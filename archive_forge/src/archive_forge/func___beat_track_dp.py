import numpy as np
import scipy
import scipy.stats
from ._cache import cache
from . import core
from . import onset
from . import util
from .feature import tempogram, fourier_tempogram
from .feature import tempo as _tempo
from .util.exceptions import ParameterError
from .util.decorators import moved
from typing import Any, Callable, Optional, Tuple
def __beat_track_dp(localscore, period, tightness):
    """Core dynamic program for beat tracking"""
    backlink = np.zeros_like(localscore, dtype=int)
    cumscore = np.zeros_like(localscore)
    window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)
    if tightness <= 0:
        raise ParameterError('tightness must be strictly positive')
    txwt = -tightness * np.log(-window / period) ** 2
    first_beat = True
    for i, score_i in enumerate(localscore):
        z_pad = np.maximum(0, min(-window[0], len(window)))
        candidates = txwt.copy()
        candidates[z_pad:] = candidates[z_pad:] + cumscore[window[z_pad:]]
        beat_location = np.argmax(candidates)
        cumscore[i] = score_i + candidates[beat_location]
        if first_beat and score_i < 0.01 * localscore.max():
            backlink[i] = -1
        else:
            backlink[i] = window[beat_location]
            first_beat = False
        window = window + 1
    return (backlink, cumscore)