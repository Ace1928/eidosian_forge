import matplotlib as mpl
import numpy as np
import param
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.dates import DateFormatter, date2num
from packaging.version import Version
from ...core.dimension import Dimension, dimension_name
from ...core.options import Store, abbreviated_exception
from ...core.util import (
from ...element import HeatMap, Raster
from ...operation import interpolate_curve
from ...util.transform import dim
from ..mixins import AreaMixin, BarsMixin, SpikesMixin
from ..plot import PlotSelector
from ..util import compute_sizes, get_min_distance, get_sideplot_ranges
from .element import ColorbarPlot, ElementPlot, LegendPlot
from .path import PathPlot
from .plot import AdjoinedPlot, mpl_rc_context
from .util import mpl_version
def _compute_ticks(self, element, edges, widths, lims):
    """
        Compute the ticks either as cyclic values in degrees or as roughly
        evenly spaced bin centers.
        """
    if self.xticks is None or not isinstance(self.xticks, int):
        return None
    if self.cyclic:
        x0, x1, _, _ = lims
        xvals = np.linspace(x0, x1, self.xticks)
        labels = [f'{np.rad2deg(x):.0f}°' for x in xvals]
    elif self.xticks:
        dim = element.get_dimension(0)
        inds = np.linspace(0, len(edges), self.xticks, dtype=int)
        edges = list(edges) + [edges[-1] + widths[-1]]
        xvals = [edges[i] for i in inds]
        labels = [dim.pprint_value(v) for v in xvals]
    return [xvals, labels]