from dataclasses import dataclass
from typing import List, Optional
from .base import BaseInferenceType
@dataclass
class ZeroShotClassificationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Zero Shot Classification
    """
    hypothesis_template: Optional[str] = None
    'The sentence used in conjunction with candidateLabels to attempt the text classification\n    by replacing the placeholder with the candidate labels.\n    '
    multi_label: Optional[bool] = None
    'Whether multiple candidate labels can be true. If false, the scores are normalized such\n    that the sum of the label likelihoods for each sequence is 1. If true, the labels are\n    considered independent and probabilities are normalized for each candidate.\n    '