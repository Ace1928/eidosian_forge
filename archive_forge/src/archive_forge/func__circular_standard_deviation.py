import warnings
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc
from .. import _log
from ..utils import conditional_jit, conditional_vect, conditional_dask
from .density_utils import histogram as _histogram
def _circular_standard_deviation(samples, high=2 * np.pi, low=0, skipna=False, axis=None):
    ang = _circfunc(samples, high, low, skipna)
    s_s = np.sin(ang).mean(axis=axis)
    c_c = np.cos(ang).mean(axis=axis)
    r_r = np.hypot(s_s, c_c)
    return (high - low) / 2.0 / np.pi * np.sqrt(-2 * np.log(r_r))