import warnings
import bokeh.plotting as bkp
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Title
from bokeh.models.glyphs import Scatter
from ....rcparams import _validate_bokeh_marker, rcParams
from ...plot_utils import _scale_fig_size, color_from_dim, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Bokeh elpd plot.