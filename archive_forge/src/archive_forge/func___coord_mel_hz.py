from __future__ import annotations
from itertools import product
import warnings
import numpy as np
import matplotlib.cm as mcm
import matplotlib.axes as mplaxes
import matplotlib.ticker as mplticker
import matplotlib.pyplot as plt
from . import core
from . import util
from .util.deprecation import rename_kw, Deprecated
from .util.exceptions import ParameterError
from typing import TYPE_CHECKING, Any, Collection, Optional, Union, Callable, Dict
from ._typing import _FloatLike_co
def __coord_mel_hz(n: int, fmin: Optional[float]=0.0, fmax: Optional[float]=None, sr: float=22050, htk: bool=False, **_kwargs: Any) -> np.ndarray:
    """Get the frequencies for Mel bins"""
    if fmin is None:
        fmin = 0.0
    if fmax is None:
        fmax = 0.5 * sr
    basis = core.mel_frequencies(n, fmin=fmin, fmax=fmax, htk=htk)
    return basis