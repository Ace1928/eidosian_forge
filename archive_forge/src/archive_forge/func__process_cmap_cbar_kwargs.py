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
def _process_cmap_cbar_kwargs(func, data, cmap=None, colors=None, cbar_kwargs: Iterable[tuple[str, Any]] | Mapping[str, Any] | None=None, levels=None, _is_facetgrid=False, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Parameters
    ----------
    func : plotting function
    data : ndarray,
        Data values

    Returns
    -------
    cmap_params : dict
    cbar_kwargs : dict
    """
    if func.__name__ == 'surface':
        kwargs['cmap'] = cmap
        return ({k: kwargs.get(k, None) for k in ['vmin', 'vmax', 'cmap', 'extend', 'levels', 'norm']}, {})
    cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)
    if 'contour' in func.__name__ and levels is None:
        levels = 7
    if cmap and colors:
        raise ValueError("Can't specify both cmap and colors.")
    if colors and ('contour' not in func.__name__ and levels is None):
        raise ValueError('Can only specify colors with contour or levels')
    if isinstance(cmap, (list, tuple)):
        raise ValueError('Specifying a list of colors in cmap is deprecated. Use colors keyword instead.')
    cmap_kwargs = {'plot_data': data, 'levels': levels, 'cmap': colors if colors else cmap, 'filled': func.__name__ != 'contour'}
    cmap_args = getfullargspec(_determine_cmap_params).args
    cmap_kwargs.update(((a, kwargs[a]) for a in cmap_args if a in kwargs))
    if not _is_facetgrid:
        cmap_params = _determine_cmap_params(**cmap_kwargs)
    else:
        cmap_params = {k: cmap_kwargs[k] for k in ['vmin', 'vmax', 'cmap', 'extend', 'levels', 'norm']}
    return (cmap_params, cbar_kwargs)