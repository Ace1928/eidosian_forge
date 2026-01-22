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
def __same_axes(x_axis, y_axis, xlim, ylim):
    """Check if two axes are similar, used to determine squared plots"""
    axes_compatible_and_not_none = (x_axis, y_axis) in _AXIS_COMPAT
    axes_same_lim = xlim == ylim
    return axes_compatible_and_not_none and axes_same_lim