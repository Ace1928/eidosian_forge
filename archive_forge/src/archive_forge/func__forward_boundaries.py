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
def _forward_boundaries(self, x):
    b = self._boundaries
    y = np.interp(x, b, np.linspace(0, 1, len(b)))
    eps = (b[-1] - b[0]) * 1e-06
    y[x < b[0] - eps] = -1
    y[x > b[-1] + eps] = 2
    return y