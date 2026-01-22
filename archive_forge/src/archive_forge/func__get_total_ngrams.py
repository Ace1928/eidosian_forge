from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _get_total_ngrams(n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]]) -> Dict[int, Tensor]:
    """Get total sum of n-grams over n-grams w.r.t n."""
    total_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
    for n in n_grams_counts:
        total_n_grams[n] = tensor(sum(n_grams_counts[n].values()))
    return total_n_grams