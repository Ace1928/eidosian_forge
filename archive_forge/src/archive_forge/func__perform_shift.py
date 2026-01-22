import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _perform_shift(words: List[str], start: int, length: int, target: int) -> List[str]:
    """Perform a shift in ``words`` from ``start`` to ``target``.

    Args:
        words: A words to shift.
        start: An index where to start shifting from.
        length: A number of how many words to be considered.
        target: An index where to end shifting.

    Return:
        A list of shifted words.

    """

    def _shift_word_before_previous_position(words: List[str], start: int, target: int, length: int) -> List[str]:
        return words[:target] + words[start:start + length] + words[target:start] + words[start + length:]

    def _shift_word_after_previous_position(words: List[str], start: int, target: int, length: int) -> List[str]:
        return words[:start] + words[start + length:target] + words[start:start + length] + words[target:]

    def _shift_word_within_shifted_string(words: List[str], start: int, target: int, length: int) -> List[str]:
        shifted_words = words[:start]
        shifted_words += words[start + length:length + target]
        shifted_words += words[start:start + length]
        shifted_words += words[length + target:]
        return shifted_words
    if target < start:
        return _shift_word_before_previous_position(words, start, target, length)
    if target > start + length:
        return _shift_word_after_previous_position(words, start, target, length)
    return _shift_word_within_shifted_string(words, start, target, length)