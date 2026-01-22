from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
def _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params):
    cbar_kwargs.setdefault('extend', cmap_params['extend'])
    if cbar_ax is None:
        cbar_kwargs.setdefault('ax', ax)
    else:
        cbar_kwargs.setdefault('cax', cbar_ax)
    if hasattr(primitive, 'extend'):
        cbar_kwargs.pop('extend')
    fig = ax.get_figure()
    cbar = fig.colorbar(primitive, **cbar_kwargs)
    return cbar