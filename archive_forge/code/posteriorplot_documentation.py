from numbers import Number
from typing import Optional
import numpy as np
from bokeh.models.annotations import Title
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram
from ...kdeplot import plot_kde
from ...plot_utils import (
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Artist to draw posterior.