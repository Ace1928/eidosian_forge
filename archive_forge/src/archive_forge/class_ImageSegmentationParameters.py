from dataclasses import dataclass
from typing import Any, Literal, Optional
from .base import BaseInferenceType
@dataclass
class ImageSegmentationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Image Segmentation
    """
    mask_threshold: Optional[float] = None
    'Threshold to use when turning the predicted masks into binary values.'
    overlap_mask_area_threshold: Optional[float] = None
    'Mask overlap threshold to eliminate small, disconnected segments.'
    subtask: Optional['ImageSegmentationSubtask'] = None
    'Segmentation task to be performed, depending on model capabilities.'
    threshold: Optional[float] = None
    'Probability threshold to filter out predicted masks.'