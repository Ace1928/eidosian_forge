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
def _inject_kwargs(self, func, kws, params):
    """Add params to kws if they are accepted by func."""
    func_params = signature(func).parameters
    for key, val in params.items():
        if key in func_params:
            kws.setdefault(key, val)