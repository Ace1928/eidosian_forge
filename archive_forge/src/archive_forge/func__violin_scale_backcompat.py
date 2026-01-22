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
def _violin_scale_backcompat(self, scale, scale_hue, density_norm, common_norm):
    """Provide two cycles of backcompat for scale kwargs"""
    if scale is not deprecated:
        density_norm = scale
        msg = f'\n\nThe `scale` parameter has been renamed and will be removed in v0.15.0. Pass `density_norm={scale!r}` for the same effect.'
        warnings.warn(msg, FutureWarning, stacklevel=3)
    if scale_hue is not deprecated:
        common_norm = scale_hue
        msg = f'\n\nThe `scale_hue` parameter has been replaced and will be removed in v0.15.0. Pass `common_norm={not scale_hue}` for the same effect.'
        warnings.warn(msg, FutureWarning, stacklevel=3)
    return (density_norm, common_norm)