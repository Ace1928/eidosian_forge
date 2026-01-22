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
def _map_diag_iter_hue(self, func, **kwargs):
    """Put marginal plot on each diagonal axes, iterating over hue."""
    fixed_color = kwargs.pop('color', None)
    for var, ax in zip(self.diag_vars, self.diag_axes):
        hue_grouped = self.data[var].groupby(self.hue_vals, observed=True)
        plot_kwargs = kwargs.copy()
        if str(func.__module__).startswith('seaborn'):
            plot_kwargs['ax'] = ax
        else:
            plt.sca(ax)
        for k, label_k in enumerate(self._hue_order):
            try:
                data_k = hue_grouped.get_group(label_k)
            except KeyError:
                data_k = pd.Series([], dtype=float)
            if fixed_color is None:
                color = self.palette[k]
            else:
                color = fixed_color
            if self._dropna:
                data_k = utils.remove_na(data_k)
            if str(func.__module__).startswith('seaborn'):
                func(x=data_k, label=label_k, color=color, **plot_kwargs)
            else:
                func(data_k, label=label_k, color=color, **plot_kwargs)
    self._add_axis_labels()
    return self