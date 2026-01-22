import numpy as np
from bokeh.models import BoxAnnotation
from bokeh.models.annotations import Title
from scipy import stats
from ....stats.density_utils import kde
from ....stats.stats_utils import smooth_data
from ...kdeplot import plot_kde
from ...plot_utils import (
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Bokeh bpv plot.