import math
import warnings
import numpy as np
from ...plot_utils import _scale_fig_size, vectorized_to_hex
from .. import show_layout
from . import create_axes_grid
from ...plot_utils import plot_point_interval
from ...dotplot import wilkinson_algorithm, layout_stacks
Bokeh dotplot.