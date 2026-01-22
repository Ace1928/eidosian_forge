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
def _find_numeric_cols(self, data):
    """Find which variables in a DataFrame are numeric."""
    numeric_cols = []
    for col in data:
        if variable_type(data[col]) == 'numeric':
            numeric_cols.append(col)
    return numeric_cols