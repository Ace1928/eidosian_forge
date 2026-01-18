import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from cryptography.fernet import Fernet

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Manage security for API keys
class SecurityManager:
    def __init__(self):
        crypto_key = os.getenv("CRYPTO_KEY")
        if not crypto_key:
            raise ValueError("CRYPTO_KEY environment variable not set.")
        self.cipher_suite = Fernet(crypto_key)

    def encrypt(self, data: str) -> str:
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt(self, data: str) -> str:
        try:
            return self.cipher_suite.decrypt(data.encode()).decode()
        except Fernet.InvalidToken:
            raise ValueError(
                "Decryption failed. Check if the encrypted data or key has changed."
            )


# Configuration management
class ConfigManager:
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.settings = {}
        self.OpenAIInterface = OpenAIInterface(api_key="")

    def update_settings(self, new_settings: dict):
        self.settings.update(new_settings)
        self.save_settings()

    def load_settings(self) -> dict:
        return {
            "api_key_encrypted": os.getenv("ENCRYPTED_API_KEY", ""),
            "model": "text-davinci-002",
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def save_settings(self):
        pass

    def encrypt_api_key(self, key: str) -> str:
        return self.security_manager.encrypt(key)

    def decrypt_api_key(self) -> str:
        encrypted_key = self.settings.get("api_key_encrypted")
        if not encrypted_key:
            raise ValueError("API key is not set.")
        return self.security_manager.decrypt(encrypted_key)

    def fetch_model_options(self):
        """Fetches and returns available model options from OpenAI."""
        try:
            models = (
                self.OpenAIInterface.fetch_model_options()
            )  # Assuming this method exists in OpenAIInterface
            model_options = [model.id for model in models]
            return model_options
        except Exception as e:
            st.error(f"Failed to fetch model options: {str(e)}")
            return []


# OpenAI API interaction
class OpenAIInterface:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_text_response(self, prompt: str, model_params: dict) -> str:
        response = client.completions.create(
            engine=model_params["model"],
            prompt=prompt,
            max_tokens=model_params["max_tokens"],
            temperature=model_params["temperature"],
            top_p=model_params["top_p"],
            frequency_penalty=model_params["frequency_penalty"],
            presence_penalty=model_params["presence_penalty"],
        )
        return response.choices[0].text.strip()

    def generate_image(self, prompt: str) -> str:
        response = client.images.generate(
            prompt=prompt, n=1
        )  # Number of images to generate
        return response.data[0].url

    def transcribe_audio(self, audio_file_path: str) -> str:
        response = client.audio.transcriptions.create(
            file=client.files.upload(file_path=audio_file_path)
        )
        return response.transcription_text

    def fetch_model_options(self) -> list:
        try:
            models = client.Model.list()
            model_options = [model.id for model in models.data]
            return model_options
        except Exception as e:
            print(f"Failed to fetch model options: {str(e)}")
            return []


# Streamlit application setup
def app():
    st.title("Advanced OpenAI Services Interface")
    security_manager = SecurityManager()
    config_manager = ConfigManager(security_manager)

    with st.sidebar:
        st.header("Configuration")
        if st.button("Load Configuration"):
            config_manager.settings = config_manager.load_settings()
        if st.button("Save Configuration"):
            config_manager.save_settings()

        api_key = st.text_input("API Key", type="password")
        if api_key:
            encrypted_api_key = config_manager.encrypt_api_key(api_key)
            config_manager.settings["api_key_encrypted"] = encrypted_api_key

        model_options = config_manager.fetch_model_options() if api_key else []
        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=(
                model_options.index(
                    config_manager.settings.get("model", "text-davinci-002")
                )
                if "model" in config_manager.settings
                else 0
            ),
        )
        config_manager.settings["model"] = selected_model

        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=2048,
            value=config_manager.settings.get("max_tokens", 150),
        )
        config_manager.settings["max_tokens"] = max_tokens

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config_manager.settings.get("temperature", 0.7),
        )
        config_manager.settings["temperature"] = temperature

        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=config_manager.settings.get("top_p", 1.0),
        )
        config_manager.settings["top_p"] = top_p

        frequency_penalty = st.slider(
            "Frequency Penalty",
            min_value=0.0,
            max_value=1.0,
            value=config_manager.settings.get("frequency_penalty", 0.0),
        )
        config_manager.settings["frequency_penalty"] = frequency_penalty

        presence_penalty = st.slider(
            "Presence Penalty",
            min_value=0.0,
            max_value=1.0,
            value=config_manager.settings.get("presence_penalty", 0.0),
        )
        config_manager.settings["presence_penalty"] = presence_penalty

    if st.button("Initialize API Interface"):
        api_interface = OpenAIInterface(config_manager.decrypt_api_key())

        # UI for different services
        service_type = st.selectbox(
            "Select Service",
            ["Text Completion", "Image Generation", "Audio Transcription"],
        )

        if service_type == "Text Completion":
            user_input = st.text_area("Type your message here:")
            if st.button("Send"):
                model_params = config_manager.settings
                response = api_interface.get_text_response(user_input, model_params)
                st.write(f"Response: {response}")
        elif service_type == "Image Generation":
            prompt = st.text_input("Image Prompt")
            if st.button("Generate"):
                image_url = api_interface.generate_image(prompt)
                st.image(image_url, caption="Generated Image")
        elif service_type == "Audio Transcription":
            audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
            if st.button("Transcribe") and audio_file:
                response_text = api_interface.transcribe_audio(audio_file)
                st.write(f"Transcription: {response_text}")


if __name__ == "__main__":
    app()
