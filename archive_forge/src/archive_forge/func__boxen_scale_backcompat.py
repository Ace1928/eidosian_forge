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
def _boxen_scale_backcompat(self, scale, width_method):
    """Provide two cycles of backcompat for scale kwargs"""
    if scale is not deprecated:
        width_method = scale
        msg = f'\n\nThe `scale` parameter has been renamed to `width_method` and will be removed in v0.15. Pass `width_method={scale!r}'
        if scale == 'area':
            msg += ", but note that the result for 'area' will appear different."
        else:
            msg += ' for the same effect.'
        warnings.warn(msg, FutureWarning, stacklevel=3)
    return width_method