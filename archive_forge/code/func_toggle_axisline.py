from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
def toggle_axisline(self, b=None):
    if b is None:
        b = not self._axisline_on
    if b:
        self._axisline_on = True
        self.spines[:].set_visible(False)
        self.xaxis.set_visible(False)
        self.yaxis.set_visible(False)
    else:
        self._axisline_on = False
        self.spines[:].set_visible(True)
        self.xaxis.set_visible(True)
        self.yaxis.set_visible(True)