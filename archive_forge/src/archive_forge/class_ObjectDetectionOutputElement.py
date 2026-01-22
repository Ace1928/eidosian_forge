from dataclasses import dataclass
from typing import Any, Optional
from .base import BaseInferenceType
@dataclass
class ObjectDetectionOutputElement(BaseInferenceType):
    """Outputs of inference for the Object Detection task"""
    box: ObjectDetectionBoundingBox
    'The predicted bounding box. Coordinates are relative to the top left corner of the input\n    image.\n    '
    label: str
    'The predicted label for the bounding box'
    score: float
    'The associated score / probability'