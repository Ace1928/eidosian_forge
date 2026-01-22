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
def _complement_color(self, color, base_color, hue_map):
    """Allow a color to be set automatically using a basis of comparison."""
    if color == 'gray':
        msg = 'Use "auto" to set automatic grayscale colors. From v0.14.0, "gray" will default to matplotlib\'s definition.'
        warnings.warn(msg, FutureWarning, stacklevel=3)
        color = 'auto'
    elif color is None or color is default:
        color = 'auto'
    if color != 'auto':
        return color
    if hue_map.lookup_table is None:
        if base_color is None:
            return None
        basis = [mpl.colors.to_rgb(base_color)]
    else:
        basis = [mpl.colors.to_rgb(c) for c in hue_map.lookup_table.values()]
    unique_colors = np.unique(basis, axis=0)
    light_vals = [rgb_to_hls(*rgb[:3])[1] for rgb in unique_colors]
    lum = min(light_vals) * 0.6
    return (lum, lum, lum)