import matplotlib.pyplot as plt
import numpy as np
from ....stats.density_utils import get_bins, histogram
from ...kdeplot import plot_kde
from ...plot_utils import (
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Add a histogram for the data to the axes.