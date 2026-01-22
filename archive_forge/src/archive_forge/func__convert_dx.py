import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
@staticmethod
def _convert_dx(dx, x0, xconv, convert):
    """
        Small helper to do logic of width conversion flexibly.

        *dx* and *x0* have units, but *xconv* has already been converted
        to unitless (and is an ndarray).  This allows the *dx* to have units
        that are different from *x0*, but are still accepted by the
        ``__add__`` operator of *x0*.
        """
    assert type(xconv) is np.ndarray
    if xconv.size == 0:
        return convert(dx)
    try:
        try:
            x0 = cbook._safe_first_finite(x0)
        except (TypeError, IndexError, KeyError):
            pass
        try:
            x = cbook._safe_first_finite(xconv)
        except (TypeError, IndexError, KeyError):
            x = xconv
        delist = False
        if not np.iterable(dx):
            dx = [dx]
            delist = True
        dx = [convert(x0 + ddx) - x for ddx in dx]
        if delist:
            dx = dx[0]
    except (ValueError, TypeError, AttributeError):
        dx = convert(dx)
    return dx