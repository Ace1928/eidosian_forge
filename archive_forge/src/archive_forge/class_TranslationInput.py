from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TranslationInput(BaseInferenceType):
    """Inputs for Translation inference
    Inputs for Text2text Generation inference
    """
    inputs: str
    'The input text data'
    parameters: Optional[TranslationGenerationParameters] = None
    'Additional inference parameters'