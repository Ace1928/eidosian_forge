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
def __check_axes(axes: Optional[mplaxes.Axes]) -> mplaxes.Axes:
    """Check if "axes" is an instance of an axis object. If not, use `gca`."""
    if axes is None:
        axes = plt.gca()
    elif not isinstance(axes, mplaxes.Axes):
        raise ParameterError('`axes` must be an instance of matplotlib.axes.Axes. Found type(axes)={}'.format(type(axes)))
    return axes