from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_community.tools.azure_ai_services.utils import (
def _speech_to_text(self, audio_path: str, speech_language: str) -> str:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        pass
    audio_src_type = detect_file_src_type(audio_path)
    if audio_src_type == 'local':
        audio_config = speechsdk.AudioConfig(filename=audio_path)
    elif audio_src_type == 'remote':
        tmp_audio_path = download_audio_from_url(audio_path)
        audio_config = speechsdk.AudioConfig(filename=tmp_audio_path)
    else:
        raise ValueError(f'Invalid audio path: {audio_path}')
    self.speech_config.speech_recognition_language = speech_language
    speech_recognizer = speechsdk.SpeechRecognizer(self.speech_config, audio_config)
    return self._continuous_recognize(speech_recognizer)