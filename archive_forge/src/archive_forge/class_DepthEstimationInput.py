from dataclasses import dataclass
from typing import Any, Dict, Optional
from .base import BaseInferenceType
@dataclass
class DepthEstimationInput(BaseInferenceType):
    """Inputs for Depth Estimation inference"""
    inputs: Any
    'The input image data'
    parameters: Optional[Dict[str, Any]] = None
    'Additional inference parameters'