from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.lib import i8max
from pandas._libs.tslibs import (
def _generate_range_overflow_safe_signed(endpoint: int, periods: int, stride: int, side: str) -> int:
    """
    A special case for _generate_range_overflow_safe where `periods * stride`
    can be calculated without overflowing int64 bounds.
    """
    assert side in ['start', 'end']
    if side == 'end':
        stride *= -1
    with np.errstate(over='raise'):
        addend = np.int64(periods) * np.int64(stride)
        try:
            result = np.int64(endpoint) + addend
            if result == iNaT:
                raise OverflowError
            return int(result)
        except (FloatingPointError, OverflowError):
            pass
        assert stride > 0 and endpoint >= 0 or (stride < 0 and endpoint <= 0)
        if stride > 0:
            uresult = np.uint64(endpoint) + np.uint64(addend)
            i64max = np.uint64(i8max)
            assert uresult > i64max
            if uresult <= i64max + np.uint64(stride):
                return int(uresult)
    raise OutOfBoundsDatetime(f'Cannot generate range with {side}={endpoint} and periods={periods}')