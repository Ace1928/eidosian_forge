import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
def add_auto_adjustable_area(self, use_axes, pad=0.1, adjust_dirs=None):
    """
        Add auto-adjustable padding around *use_axes* to take their decorations
        (title, labels, ticks, ticklabels) into account during layout.

        Parameters
        ----------
        use_axes : `~matplotlib.axes.Axes` or list of `~matplotlib.axes.Axes`
            The Axes whose decorations are taken into account.
        pad : float, default: 0.1
            Additional padding in inches.
        adjust_dirs : list of {"left", "right", "bottom", "top"}, optional
            The sides where padding is added; defaults to all four sides.
        """
    if adjust_dirs is None:
        adjust_dirs = ['left', 'right', 'bottom', 'top']
    for d in adjust_dirs:
        self.append_size(d, Size._AxesDecorationsSize(use_axes, d) + pad)