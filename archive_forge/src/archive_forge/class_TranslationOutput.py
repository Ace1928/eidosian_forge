from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TranslationOutput(BaseInferenceType):
    """Outputs of inference for the Translation task"""
    translation_text: str
    'The translated text.'