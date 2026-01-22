from numbers import Number
from functools import partial
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from ._base import VectorPlotter
from ._statistics import ECDF, Histogram, KDE
from ._stats.counting import Hist
from .axisgrid import (
from .utils import (
from .palettes import color_palette
from .external import husl
from .external.kde import gaussian_kde
from ._docstrings import (
important parameter. Misspecification of the bandwidth can produce a
def _default_discrete(self):
    """Find default values for discrete hist estimation based on variable type."""
    if self.univariate:
        discrete = self.var_types[self.data_variable] == 'categorical'
    else:
        discrete_x = self.var_types['x'] == 'categorical'
        discrete_y = self.var_types['y'] == 'categorical'
        discrete = (discrete_x, discrete_y)
    return discrete