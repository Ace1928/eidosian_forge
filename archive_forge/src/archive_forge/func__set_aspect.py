import copy
import math
import warnings
from types import FunctionType
import matplotlib.colors as mpl_colors
import numpy as np
import param
from matplotlib import ticker
from matplotlib.dates import date2num
from matplotlib.image import AxesImage
from packaging.version import Version
from ...core import (
from ...core.options import Keywords, abbreviated_exception
from ...element import Graph, Path
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_range_key, process_cmap
from .plot import MPLPlot, mpl_rc_context
from .util import EqHistNormalize, mpl_version, validate, wrap_formatter
def _set_aspect(self, axes, aspect):
    """
        Set the aspect on the axes based on the aspect setting.
        """
    if isinstance(self.projection, str) and self.projection == '3d':
        return
    if isinstance(aspect, str) and aspect != 'square' or self.data_aspect:
        data_ratio = self.data_aspect or aspect
    else:
        (x0, x1), (y0, y1) = (axes.get_xlim(), axes.get_ylim())
        xsize = np.log(x1) - np.log(x0) if self.logx else x1 - x0
        ysize = np.log(y1) - np.log(y0) if self.logy else y1 - y0
        xsize = max(abs(xsize), 1e-30)
        ysize = max(abs(ysize), 1e-30)
        data_ratio = 1.0 / (ysize / xsize)
        if aspect != 'square':
            data_ratio = data_ratio / aspect
    axes.set_aspect(data_ratio)