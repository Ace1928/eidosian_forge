from dataclasses import dataclass
from typing import Any, List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TokenClassificationInput(BaseInferenceType):
    """Inputs for Token Classification inference"""
    inputs: str
    'The input text data'
    parameters: Optional[TokenClassificationParameters] = None
    'Additional inference parameters'