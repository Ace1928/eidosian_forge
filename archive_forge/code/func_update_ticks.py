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
def update_ticks(self):
    """
        Set up the ticks and ticklabels. This should not be needed by users.
        """
    self._get_ticker_locator_formatter()
    self._long_axis().set_major_locator(self._locator)
    self._long_axis().set_minor_locator(self._minorlocator)
    self._long_axis().set_major_formatter(self._formatter)