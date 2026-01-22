from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class TextToImageInput(BaseInferenceType):
    """Inputs for Text To Image inference"""
    inputs: str
    'The input text data (sometimes called "prompt"'
    parameters: Optional[TextToImageParameters] = None
    'Additional inference parameters'