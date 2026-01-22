import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple, Union
from torch import Tensor, tensor
from torchmetrics.utilities import rank_zero_warn
def _compute_exact_match_score(prediction: str, ground_truth: str) -> Tensor:
    """Compute Exact Match for two sentences."""
    return tensor(int(_normalize_text(prediction) == _normalize_text(ground_truth)))