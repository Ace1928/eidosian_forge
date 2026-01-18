import warnings
from collections.abc import Iterable
from itertools import cycle
import bokeh.plotting as bkp
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import ColumnDataSource, DataRange1d, Span
from bokeh.models.glyphs import Scatter
from bokeh.models.annotations import Title
from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size
from ...rankplot import plot_rank
from .. import show_layout
from . import backend_kwarg_defaults, dealiase_sel_kwargs
from ....sel_utils import xarray_var_iter
Bokeh traceplot.