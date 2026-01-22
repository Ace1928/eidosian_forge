import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _translation_edit_rate(pred_words: List[str], target_words: List[str]) -> Tensor:
    """Compute translation edit rate between hypothesis and reference sentences.

    Args:
        pred_words: A list of a tokenized hypothesis sentence.
        target_words: A list of lists of tokenized reference sentences.

    Return:
        A number of required edits to match hypothesis and reference sentences.

    """
    if len(target_words) == 0:
        return tensor(0.0)
    cached_edit_distance = _LevenshteinEditDistance(target_words)
    num_shifts = 0
    checked_candidates = 0
    input_words = pred_words
    while True:
        delta, new_input_words, checked_candidates = _shift_words(input_words, target_words, cached_edit_distance, checked_candidates)
        if checked_candidates >= _MAX_SHIFT_CANDIDATES or delta <= 0:
            break
        num_shifts += 1
        input_words = new_input_words
    edit_distance, _ = cached_edit_distance(input_words)
    total_edits = num_shifts + edit_distance
    return tensor(total_edits)