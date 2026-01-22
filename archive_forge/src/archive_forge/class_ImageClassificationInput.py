from dataclasses import dataclass
from typing import Any, Literal, Optional
from .base import BaseInferenceType
@dataclass
class ImageClassificationInput(BaseInferenceType):
    """Inputs for Image Classification inference"""
    inputs: Any
    'The input image data'
    parameters: Optional[ImageClassificationParameters] = None
    'Additional inference parameters'