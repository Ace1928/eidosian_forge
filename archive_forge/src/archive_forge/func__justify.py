from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def _justify(head: list[Sequence[str]], tail: list[Sequence[str]]) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """
    Justify items in head and tail, so they are right-aligned when stacked.

    Parameters
    ----------
    head : list-like of list-likes of strings
    tail : list-like of list-likes of strings

    Returns
    -------
    tuple of list of tuples of strings
        Same as head and tail, but items are right aligned when stacked
        vertically.

    Examples
    --------
    >>> _justify([['a', 'b']], [['abc', 'abcd']])
    ([('  a', '   b')], [('abc', 'abcd')])
    """
    combined = head + tail
    max_length = [0] * len(combined[0])
    for inner_seq in combined:
        length = [len(item) for item in inner_seq]
        max_length = [max(x, y) for x, y in zip(max_length, length)]
    head_tuples = [tuple((x.rjust(max_len) for x, max_len in zip(seq, max_length))) for seq in head]
    tail_tuples = [tuple((x.rjust(max_len) for x, max_len in zip(seq, max_length))) for seq in tail]
    return (head_tuples, tail_tuples)