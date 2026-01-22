from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings
import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval
def _ecdf_right_censored(sample):
    tod = sample._uncensored
    tol = sample._right
    times = np.concatenate((tod, tol))
    died = np.asarray([1] * tod.size + [0] * tol.size)
    i = np.argsort(times)
    times = times[i]
    died = died[i]
    at_risk = np.arange(times.size, 0, -1)
    j = np.diff(times, prepend=-np.inf, append=np.inf) > 0
    j_l = j[:-1]
    j_r = j[1:]
    t = times[j_l]
    n = at_risk[j_l]
    cd = np.cumsum(died)[j_r]
    d = np.diff(cd, prepend=0)
    sf = np.cumprod((n - d) / n)
    cdf = 1 - sf
    return (t, cdf, sf, n, d)