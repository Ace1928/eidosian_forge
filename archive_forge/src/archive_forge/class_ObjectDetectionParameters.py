from dataclasses import dataclass
from typing import Any, Optional
from .base import BaseInferenceType
@dataclass
class ObjectDetectionParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Object Detection
    """
    threshold: Optional[float] = None
    'The probability necessary to make a prediction.'