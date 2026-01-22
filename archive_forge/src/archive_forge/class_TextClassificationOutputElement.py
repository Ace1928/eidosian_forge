from dataclasses import dataclass
from typing import Literal, Optional
from .base import BaseInferenceType
@dataclass
class TextClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Text Classification task"""
    label: str
    'The predicted class label.'
    score: float
    'The corresponding probability.'