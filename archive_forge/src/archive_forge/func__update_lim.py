import matplotlib as mpl
import numpy as np
import pandas as pd
import param
from matplotlib import patches
from matplotlib.lines import Line2D
from ...core.options import abbreviated_exception
from ...core.util import match_spec
from ...element import HLines, HSpans, VLines, VSpans
from .element import ColorbarPlot, ElementPlot
from .plot import mpl_rc_context
def _update_lim(self, event):
    """ called whenever axis x/y limits change """
    x = np.array(self.axes.get_xbound())
    y = self._slope * x + self._intercept
    self.set_data(x, y)
    self.axes.draw_artist(self)