from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class AutomaticSpeechRecognitionInput(BaseInferenceType):
    """Inputs for Automatic Speech Recognition inference"""
    inputs: Any
    'The input audio data'
    parameters: Optional[AutomaticSpeechRecognitionParameters] = None
    'Additional inference parameters'