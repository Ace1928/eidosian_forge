from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import (
from .palettes import color_palette, blend_palette
from ._docstrings import (
@property
def _not_left_axes(self):
    """Return a flat array of axes that aren't on the left column."""
    if self._col_wrap is None:
        return self.axes[:, 1:].flat
    else:
        axes = []
        for i, ax in enumerate(self.axes):
            if i % self._ncol:
                axes.append(ax)
        return np.array(axes, object).flat