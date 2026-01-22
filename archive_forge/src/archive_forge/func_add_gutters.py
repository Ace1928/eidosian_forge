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
def add_gutters(self, points, center, trans_fwd, trans_inv):
    """Stop points from extending beyond their territory."""
    half_width = self.width / 2
    low_gutter = trans_inv(trans_fwd(center) - half_width)
    off_low = points < low_gutter
    if off_low.any():
        points[off_low] = low_gutter
    high_gutter = trans_inv(trans_fwd(center) + half_width)
    off_high = points > high_gutter
    if off_high.any():
        points[off_high] = high_gutter
    gutter_prop = (off_high + off_low).sum() / len(points)
    if gutter_prop > self.warn_thresh:
        msg = '{:.1%} of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.'.format(gutter_prop)
        warnings.warn(msg, UserWarning)
    return points