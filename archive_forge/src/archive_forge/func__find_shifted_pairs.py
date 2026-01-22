import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _find_shifted_pairs(pred_words: List[str], target_words: List[str]) -> Iterator[Tuple[int, int, int]]:
    """Find matching word sub-sequences in two lists of words. Ignores sub- sequences starting at the same position.

    Args:
        pred_words: A list of a tokenized hypothesis sentence.
        target_words: A list of a tokenized reference sentence.

    Return:
        Yields tuples of ``target_start, pred_start, length`` such that:
        ``target_words[target_start : target_start + length] == pred_words[pred_start : pred_start + length]``

        pred_start:
            A list of hypothesis start indices.
        target_start:
            A list of reference start indices.
        length:
            A length of a word span to be considered.

    """
    for pred_start in range(len(pred_words)):
        for target_start in range(len(target_words)):
            if abs(target_start - pred_start) > _MAX_SHIFT_DIST:
                continue
            for length in range(1, _MAX_SHIFT_SIZE):
                if pred_words[pred_start + length - 1] != target_words[target_start + length - 1]:
                    break
                yield (pred_start, target_start, length)
                _hyp = len(pred_words) == pred_start + length
                _ref = len(target_words) == target_start + length
                if _hyp or _ref:
                    break