from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _get_ngram_matches(hyp_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]], ref_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]]) -> Dict[int, Tensor]:
    """Get a number of n-gram matches between reference and hypothesis n-grams.

    Args:
        hyp_n_grams_counts: n-grams counts for hypothesis
        ref_n_grams_counts: n-grams counts for reference

    Return:
        matching_n_grams

    """
    matching_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
    for n in hyp_n_grams_counts:
        matching_n_grams[n] = tensor(sum((torch.min(ref_n_grams_counts[n][n_gram], hyp_n_grams_counts[n][n_gram]) for n_gram in hyp_n_grams_counts[n])))
    return matching_n_grams