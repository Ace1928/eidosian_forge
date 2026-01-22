from __future__ import annotations
import itertools
import warnings
import numpy as np
from numpy.typing import ArrayLike
from pandas import Series
import matplotlib as mpl
from matplotlib.colors import to_rgb, to_rgba, to_rgba_array
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from seaborn._core.scales import Scale, Boolean, Continuous, Nominal, Temporal
from seaborn._core.rules import categorical_order, variable_type
from seaborn.palettes import QUAL_PALETTES, color_palette, blend_palette
from seaborn.utils import get_color_cycle
from typing import Any, Callable, Tuple, List, Union, Optional
def _standardize_color_sequence(self, colors: ArrayLike) -> ArrayLike:
    """Convert color sequence to RGB(A) array, preserving but not adding alpha."""

    def has_alpha(x):
        return to_rgba(x) != to_rgba(x, 1)
    if isinstance(colors, np.ndarray):
        needs_alpha = colors.shape[1] == 4
    else:
        needs_alpha = any((has_alpha(x) for x in colors))
    if needs_alpha:
        return to_rgba_array(colors)
    else:
        return to_rgba_array(colors)[:, :3]