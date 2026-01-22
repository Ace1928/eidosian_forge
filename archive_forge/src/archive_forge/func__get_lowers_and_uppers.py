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
def _get_lowers_and_uppers(self):
    """
        Return ``(lowers, uppers)`` for filled contours.
        """
    lowers = self._levels[:-1]
    if self.zmin == lowers[0]:
        lowers = lowers.copy()
        if self.logscale:
            lowers[0] = 0.99 * self.zmin
        else:
            lowers[0] -= 1
    uppers = self._levels[1:]
    return (lowers, uppers)