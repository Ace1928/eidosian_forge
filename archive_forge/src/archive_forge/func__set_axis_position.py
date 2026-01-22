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
def _set_axis_position(self, axes, axis, option):
    """
        Set the position and visibility of the xaxis or yaxis by
        supplying the axes object, the axis to set, i.e. 'x' or 'y'
        and an option to specify the position and visibility of the axis.
        The option may be None, 'bare' or positional, i.e. 'left' and
        'right' for the yaxis and 'top' and 'bottom' for the xaxis.
        May also combine positional and 'bare' into for example 'left-bare'.
        """
    positions = {'x': ['bottom', 'top'], 'y': ['left', 'right']}[axis]
    axis = axes.xaxis if axis == 'x' else axes.yaxis
    if option in [None, False]:
        axis.set_visible(False)
        for pos in positions:
            axes.spines[pos].set_visible(False)
    else:
        if option is True:
            option = positions[0]
        if 'bare' in option:
            axis.set_ticklabels([])
            axis.set_label_text('')
        if option != 'bare':
            option = option.split('-')[0]
            axis.set_ticks_position(option)
            axis.set_label_position(option)
    if not self.overlaid and (not self.show_frame) and (self.projection != 'polar'):
        pos = positions[1] if option and (option == 'bare' or positions[0] in option) else positions[0]
        axes.spines[pos].set_visible(False)