from collections.abc import Callable
from numbers import Integral
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.models.glyphs import Scatter
from matplotlib import colormaps
from matplotlib.colors import rgb2hex
from matplotlib.pyplot import rcParams as mpl_rcParams
from ...plot_utils import _scale_fig_size, _init_kwargs_dict
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Bokeh kde plot.