import math
import numpy as np
from numpy import ma
from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms
def _dots_per_unit(self, units):
    """Return a scale factor for converting from units to pixels."""
    bb = self.axes.bbox
    vl = self.axes.viewLim
    return _api.check_getitem({'x': bb.width / vl.width, 'y': bb.height / vl.height, 'xy': np.hypot(*bb.size) / np.hypot(*vl.size), 'width': bb.width, 'height': bb.height, 'dots': 1.0, 'inches': self.axes.figure.dpi}, units=units)