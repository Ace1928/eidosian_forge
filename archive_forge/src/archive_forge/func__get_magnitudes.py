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
def _get_magnitudes(self, element, style, ranges):
    size_dim = element.get_dimension(self.size_index)
    mag_dim = self.magnitude
    if size_dim and mag_dim:
        self.param.warning("Cannot declare style mapping for 'magnitude' option and declare a size_index; ignoring the size_index.")
    elif size_dim:
        mag_dim = size_dim
    elif isinstance(mag_dim, str):
        mag_dim = element.get_dimension(mag_dim)
    if mag_dim is not None:
        if isinstance(mag_dim, dim):
            magnitudes = mag_dim.apply(element, flat=True)
        else:
            magnitudes = element.dimension_values(mag_dim)
            _, max_magnitude = ranges[dimension_name(mag_dim)]['combined']
            if self.normalize_lengths and max_magnitude != 0:
                magnitudes = magnitudes / max_magnitude
    else:
        magnitudes = np.ones(len(element))
    return magnitudes