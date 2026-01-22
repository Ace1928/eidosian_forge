from itertools import cycle
import numpy as np
from bokeh.models import Label
from bokeh.models.annotations import Legend
from matplotlib.pyplot import rcParams as mpl_rcParams
from ....stats import bfmi as e_bfmi
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
from .distplot import _histplot_bokeh_op
Bokeh energy plot.