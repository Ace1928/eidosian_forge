import numpy as np
from bokeh.models import BoxAnnotation
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb
from xarray import DataArray
from ....stats.density_utils import kde
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Bokeh loo pit plot.