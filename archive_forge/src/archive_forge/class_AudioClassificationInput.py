from dataclasses import dataclass
from typing import Any, Literal, Optional
from .base import BaseInferenceType
@dataclass
class AudioClassificationInput(BaseInferenceType):
    """Inputs for Audio Classification inference"""
    inputs: Any
    'The input audio data'
    parameters: Optional[AudioClassificationParameters] = None
    'Additional inference parameters'