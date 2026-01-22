from dataclasses import dataclass
from typing import List, Optional
from .base import BaseInferenceType
@dataclass
class ZeroShotClassificationInput(BaseInferenceType):
    """Inputs for Zero Shot Classification inference"""
    inputs: ZeroShotClassificationInputData
    'The input text data, with candidate labels'
    parameters: Optional[ZeroShotClassificationParameters] = None
    'Additional inference parameters'