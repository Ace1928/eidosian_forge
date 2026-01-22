from __future__ import annotations
import logging
import tempfile
from typing import Any, Dict, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
def _text_to_speech(self, text: str, speech_language: str) -> str:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        pass
    self.speech_config.speech_synthesis_language = speech_language
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
    result = speech_synthesizer.speak_text(text)
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        stream = speechsdk.AudioDataStream(result)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as f:
            stream.save_to_wav_file(f.name)
        return f.name
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        logger.debug(f'Speech synthesis canceled: {cancellation_details.reason}')
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            raise RuntimeError(f'Speech synthesis error: {cancellation_details.error_details}')
        return 'Speech synthesis canceled.'
    else:
        return f'Speech synthesis failed: {result.reason}'