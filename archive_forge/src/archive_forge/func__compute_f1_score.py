import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple, Union
from torch import Tensor, tensor
from torchmetrics.utilities import rank_zero_warn
def _compute_f1_score(predicted_answer: str, target_answer: str) -> Tensor:
    """Compute F1 Score for two sentences."""
    target_tokens = _get_tokens(target_answer)
    predicted_tokens = _get_tokens(predicted_answer)
    common = Counter(target_tokens) & Counter(predicted_tokens)
    num_same = tensor(sum(common.values()))
    if len(target_tokens) == 0 or len(predicted_tokens) == 0:
        return tensor(int(target_tokens == predicted_tokens))
    if num_same == 0:
        return tensor(0.0)
    precision = 1.0 * num_same / tensor(len(predicted_tokens))
    recall = 1.0 * num_same / tensor(len(target_tokens))
    return 2 * precision * recall / (precision + recall)