from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
def _add_axes_internal(self, ax, key):
    """Private helper for `add_axes` and `add_subplot`."""
    self._axstack.add(ax)
    if ax not in self._localaxes:
        self._localaxes.append(ax)
    self.sca(ax)
    ax._remove_method = self.delaxes
    ax._projection_init = key
    self.stale = True
    ax.stale_callback = _stale_figure_callback
    return ax