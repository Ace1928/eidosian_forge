from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class ZeroShotImageClassificationOutputElement(BaseInferenceType):
    """Outputs of inference for the Zero Shot Image Classification task"""
    label: str
    'The predicted class label.'
    score: float
    'The corresponding probability.'