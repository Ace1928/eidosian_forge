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
def _check_list_length(self, levels: list, values: list) -> list:
    """Input check when values are provided as a list."""
    message = ''
    if len(levels) > len(values):
        message = ' '.join([f'\nThe {self.variable} list has fewer values ({len(values)})', f'than needed ({len(levels)}) and will cycle, which may', 'produce an uninterpretable plot.'])
        values = [x for _, x in zip(levels, itertools.cycle(values))]
    elif len(values) > len(levels):
        message = ' '.join([f'The {self.variable} list has more values ({len(values)})', f'than needed ({len(levels)}), which may not be intended.'])
        values = values[:len(levels)]
    if message:
        warnings.warn(message, UserWarning)
    return values