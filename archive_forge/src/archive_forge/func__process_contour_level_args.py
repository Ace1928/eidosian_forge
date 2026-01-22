from contextlib import ExitStack
import functools
import math
from numbers import Integral
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
def _process_contour_level_args(self, args, z_dtype):
    """
        Determine the contour levels and store in self.levels.
        """
    if self.levels is None:
        if args:
            levels_arg = args[0]
        elif np.issubdtype(z_dtype, bool):
            if self.filled:
                levels_arg = [0, 0.5, 1]
            else:
                levels_arg = [0.5]
        else:
            levels_arg = 7
    else:
        levels_arg = self.levels
    if isinstance(levels_arg, Integral):
        self.levels = self._autolev(levels_arg)
    else:
        self.levels = np.asarray(levels_arg, np.float64)
    if self.filled and len(self.levels) < 2:
        raise ValueError('Filled contours require at least 2 levels.')
    if len(self.levels) > 1 and np.min(np.diff(self.levels)) <= 0.0:
        raise ValueError('Contour levels must be increasing')