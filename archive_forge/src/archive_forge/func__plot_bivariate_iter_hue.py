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
def _plot_bivariate_iter_hue(self, x_var, y_var, ax, func, **kwargs):
    """Draw a bivariate plot while iterating over hue subsets."""
    kwargs = kwargs.copy()
    if str(func.__module__).startswith('seaborn'):
        kwargs['ax'] = ax
    else:
        plt.sca(ax)
    if x_var == y_var:
        axes_vars = [x_var]
    else:
        axes_vars = [x_var, y_var]
    hue_grouped = self.data.groupby(self.hue_vals, observed=True)
    for k, label_k in enumerate(self._hue_order):
        kws = kwargs.copy()
        try:
            data_k = hue_grouped.get_group(label_k)
        except KeyError:
            data_k = pd.DataFrame(columns=axes_vars, dtype=float)
        if self._dropna:
            data_k = data_k[axes_vars].dropna()
        x = data_k[x_var]
        y = data_k[y_var]
        for kw, val_list in self.hue_kws.items():
            kws[kw] = val_list[k]
        kws.setdefault('color', self.palette[k])
        if self._hue_var is not None:
            kws['label'] = label_k
        if str(func.__module__).startswith('seaborn'):
            func(x=x, y=y, **kws)
        else:
            func(x, y, **kws)
    self._update_legend_data(ax)