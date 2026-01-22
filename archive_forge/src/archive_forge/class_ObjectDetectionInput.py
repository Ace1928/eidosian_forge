from dataclasses import dataclass
from typing import Any, Optional
from .base import BaseInferenceType
@dataclass
class ObjectDetectionInput(BaseInferenceType):
    """Inputs for Object Detection inference"""
    inputs: Any
    'The input image data'
    parameters: Optional[ObjectDetectionParameters] = None
    'Additional inference parameters'