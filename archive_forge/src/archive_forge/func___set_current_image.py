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
def __set_current_image(ax, img):
    """
    Set the current image when working in pyplot mode.

    If the provided ``ax`` is not `None`, then we assume that the user is using the object API.
    In this case, the pyplot current image is not set.
    """
    if ax is None:
        plt.sci(img)