from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class FillMaskInput(BaseInferenceType):
    """Inputs for Fill Mask inference"""
    inputs: str
    'The text with masked tokens'
    parameters: Optional[FillMaskParameters] = None
    'Additional inference parameters'