from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
@dataclass
class label_scientific:
    """
    Scientific number labels

    Parameters
    ----------
    digits : int
        Significant digits.

    Examples
    --------
    >>> x = [.12, .23, .34, 45]
    >>> label_scientific()(x)
    ['1.2e-01', '2.3e-01', '3.4e-01', '4.5e+01']

    Notes
    -----
    Be careful when using many digits (15+ on a 64
    bit computer). Consider of the `machine epsilon`_.

    .. _machine epsilon: https://en.wikipedia.org/wiki/Machine_epsilon
    """
    digits: int = 3

    def __post_init__(self):
        tpl = f'{{:.{self.digits}e}}'
        self._label = label_custom(tpl)
        self.trailling_zeros_pattern = re.compile('(0+)e')

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        if len(x) == 0:
            return []

        def count_zeros(s):
            match = self.trailling_zeros_pattern.search(s)
            if match:
                return len(match.group(1))
            else:
                return 0
        labels = self._label(x)
        n = min([count_zeros(val) for val in labels])
        if n:
            labels = [val.replace('0' * n + 'e', 'e') for val in labels]
        return labels