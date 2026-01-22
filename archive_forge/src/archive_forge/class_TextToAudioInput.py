from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class TextToAudioInput(BaseInferenceType):
    """Inputs for Text To Audio inference"""
    inputs: str
    'The input text data'
    parameters: Optional[TextToAudioParameters] = None
    'Additional inference parameters'