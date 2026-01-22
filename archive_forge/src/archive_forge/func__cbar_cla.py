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
def _cbar_cla(self):
    """Function to clear the interactive colorbar state."""
    for x in self._interactive_funcs:
        delattr(self.ax, x)
    del self.ax.cla
    self.ax.cla()