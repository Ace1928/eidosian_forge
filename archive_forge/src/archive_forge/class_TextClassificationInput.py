from dataclasses import dataclass
from typing import Literal, Optional
from .base import BaseInferenceType
@dataclass
class TextClassificationInput(BaseInferenceType):
    """Inputs for Text Classification inference"""
    inputs: str
    'The text to classify'
    parameters: Optional[TextClassificationParameters] = None
    'Additional inference parameters'