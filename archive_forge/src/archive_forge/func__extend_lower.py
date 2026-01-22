import logging
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring
def _extend_lower(self):
    """Return whether the lower limit is open ended."""
    minmax = 'max' if self._long_axis().get_inverted() else 'min'
    return self.extend in ('both', minmax)