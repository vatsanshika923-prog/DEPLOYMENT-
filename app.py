from flask import Flask, request, jsonify
import numpy as np
import joblib  # or pickle
import traceback

app = Flask(__name__)

# Load your trained model
try:
    model = joblib.load("lstm_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

@app.route('/')
def home():
    return "LSTM (Pickle) Model API is running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input']).reshape(1, -1)
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
