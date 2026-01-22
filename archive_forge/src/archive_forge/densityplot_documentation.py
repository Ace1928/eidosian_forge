from collections import defaultdict
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models.annotations import Legend, Title
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ...plot_utils import _scale_fig_size, calculate_point_estimate, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Bokeh density plot.