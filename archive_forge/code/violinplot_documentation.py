import numpy as np
from bokeh.models.annotations import Title
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Auxiliary function to plot discrete-violinplots.