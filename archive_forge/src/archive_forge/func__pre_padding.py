from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@cache
def _pre_padding(self) -> tuple[int, int]:
    """Smallest signal index and slice index due to padding.

         Since, per convention, for time t=0, n,q is zero, the returned values
         are negative or zero.
         """
    w2 = self.win.real ** 2 + self.win.imag ** 2
    n0 = -self.m_num_mid
    for q_, n_ in enumerate(range(n0, n0 - self.m_num - 1, -self.hop)):
        n_next = n_ - self.hop
        if n_next + self.m_num <= 0 or all(w2[n_next:] == 0):
            return (n_, -q_)
    raise RuntimeError('This is code line should not have been reached!')