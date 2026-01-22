from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@lru_cache(maxsize=256)
def _post_padding(self, n: int) -> tuple[int, int]:
    """Largest signal index and slice index due to padding."""
    w2 = self.win.real ** 2 + self.win.imag ** 2
    q1 = n // self.hop
    k1 = q1 * self.hop - self.m_num_mid
    for q_, k_ in enumerate(range(k1, n + self.m_num, self.hop), start=q1):
        n_next = k_ + self.hop
        if n_next >= n or all(w2[:n - n_next] == 0):
            return (k_ + self.m_num, q_ + 1)
    raise RuntimeError('This is code line should not have been reached!')