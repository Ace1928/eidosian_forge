from __future__ import annotations
import os
from io import BytesIO
from speech_recognition.audio import AudioData
from speech_recognition.exceptions import SetupError

    Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the OpenAI Whisper API.

    This function requires an OpenAI account; visit https://platform.openai.com/signup, then generate API Key in `User settings <https://platform.openai.com/account/api-keys>`__.

    Detail: https://platform.openai.com/docs/guides/speech-to-text

    Raises a ``speech_recognition.exceptions.SetupError`` exception if there are any issues with the openai installation, or the environment variable is missing.
    