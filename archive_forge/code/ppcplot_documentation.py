import numpy as np
from bokeh.models.annotations import Legend
from bokeh.models.glyphs import Scatter
from bokeh.models import ColumnDataSource
from ....stats.density_utils import get_bins, histogram, kde
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Compute empirical cdf of a numpy array.

    Parameters
    ----------
    data : np.array
        1d array

    Returns
    -------
    np.array, np.array
        x and y coordinates for the empirical cdf of the data
    