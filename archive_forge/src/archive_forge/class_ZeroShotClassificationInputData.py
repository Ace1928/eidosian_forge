from dataclasses import dataclass
from typing import List, Optional
from .base import BaseInferenceType
@dataclass
class ZeroShotClassificationInputData(BaseInferenceType):
    """The input text data, with candidate labels"""
    candidate_labels: List[str]
    'The set of possible class labels to classify the text into.'
    text: str
    'The text to classify'