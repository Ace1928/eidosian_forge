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
def __coord_chroma(n: int, bins_per_octave: int=12, **_kwargs: Any) -> np.ndarray:
    """Get chroma bin numbers"""
    return np.linspace(0, 12.0 * n / bins_per_octave, num=n, endpoint=False)