from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
@property
def _native_width(self):
    """Return unit of width separating categories on native numeric scale."""
    if self.var_types[self.orient] == 'categorical':
        return 1
    unique_values = np.unique(self.comp_data[self.orient])
    if len(unique_values) > 1:
        native_width = np.nanmin(np.diff(unique_values))
    else:
        native_width = 1
    return native_width