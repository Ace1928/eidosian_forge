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
def _colorize_bars(self, cmap, bars, element, main_range, dim):
    """
        Use the given cmap to color the bars, applying the correct
        color ranges as necessary.
        """
    cmap_range = main_range[1] - main_range[0]
    lower_bound = main_range[0]
    colors = np.array(element.dimension_values(dim))
    colors = (colors - lower_bound) / cmap_range
    for c, bar in zip(colors, bars):
        bar.set_facecolor(cmap(c))
        bar.set_clip_on(False)