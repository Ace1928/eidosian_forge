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
def _default_values(self, n: int) -> list:
    """Return a list of n values, alternating True and False."""
    if n > 2:
        msg = ' '.join([f'The variable assigned to {self.variable} has more than two levels,', f'so {self.variable} values will cycle and may be uninterpretable'])
        warnings.warn(msg, UserWarning)
    return [x for x, _ in zip(itertools.cycle([True, False]), range(n))]