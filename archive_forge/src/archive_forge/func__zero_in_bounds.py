import math
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine
def _zero_in_bounds(self):
    """
        Return True if zero is within the valid values for the
        scale of the radial axis.
        """
    vmin, vmax = self._axes.yaxis._scale.limit_range_for_scale(0, 1, 1e-05)
    return vmin == 0