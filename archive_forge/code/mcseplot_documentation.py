import numpy as np
from bokeh.models import ColumnDataSource, Span
from bokeh.models.glyphs import Scatter
from bokeh.models.annotations import Title
from scipy.stats import rankdata
from ....stats.stats_utils import quantile as _quantile
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Bokeh mcse plot.