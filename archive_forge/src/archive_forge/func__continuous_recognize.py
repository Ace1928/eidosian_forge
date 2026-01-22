from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_community.tools.azure_cognitive_services.utils import (
def _continuous_recognize(self, speech_recognizer: Any) -> str:
    done = False
    text = ''

    def stop_cb(evt: Any) -> None:
        """callback that stop continuous recognition"""
        speech_recognizer.stop_continuous_recognition_async()
        nonlocal done
        done = True

    def retrieve_cb(evt: Any) -> None:
        """callback that retrieves the intermediate recognition results"""
        nonlocal text
        text += evt.result.text
    speech_recognizer.recognized.connect(retrieve_cb)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)
    speech_recognizer.start_continuous_recognition_async()
    while not done:
        time.sleep(0.5)
    return text