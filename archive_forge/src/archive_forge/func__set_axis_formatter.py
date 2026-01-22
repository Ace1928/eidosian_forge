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
def _set_axis_formatter(self, axis, dim, formatter):
    """
        Set axis formatter based on dimension formatter.
        """
    if isinstance(dim, list):
        dim = dim[0]
    if formatter is not None or dim is None:
        pass
    elif dim.value_format:
        formatter = dim.value_format
    elif dim.type in dim.type_formatters:
        formatter = dim.type_formatters[dim.type]
    if formatter:
        axis.set_major_formatter(wrap_formatter(formatter))