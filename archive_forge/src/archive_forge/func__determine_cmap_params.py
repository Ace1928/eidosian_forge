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
def _determine_cmap_params(plot_data, vmin=None, vmax=None, cmap=None, center=None, robust=False, extend=None, levels=None, filled=True, norm=None, _is_facetgrid=False):
    """
    Use some heuristics to set good defaults for colorbar and range.

    Parameters
    ----------
    plot_data : Numpy array
        Doesn't handle xarray objects

    Returns
    -------
    cmap_params : dict
        Use depends on the type of the plotting function
    """
    import matplotlib as mpl
    if isinstance(levels, Iterable):
        levels = sorted(levels)
    calc_data = np.ravel(plot_data[np.isfinite(plot_data)])
    if calc_data.size == 0:
        calc_data = np.array(0.0)
    possibly_divergent = center is not False
    center_is_none = False
    if center is None:
        center = 0
        center_is_none = True
    if vmin is not None and vmax is not None:
        possibly_divergent = False
    user_minmax = vmin is not None or vmax is not None
    vlim = None
    vmin_was_none = vmin is None
    vmax_was_none = vmax is None
    if vmin is None:
        if robust:
            vmin = np.percentile(calc_data, ROBUST_PERCENTILE)
        else:
            vmin = calc_data.min()
    elif possibly_divergent:
        vlim = abs(vmin - center)
    if vmax is None:
        if robust:
            vmax = np.percentile(calc_data, 100 - ROBUST_PERCENTILE)
        else:
            vmax = calc_data.max()
    elif possibly_divergent:
        vlim = abs(vmax - center)
    if possibly_divergent:
        levels_are_divergent = isinstance(levels, Iterable) and levels[0] * levels[-1] < 0
        divergent = vmin < 0 and vmax > 0 or not center_is_none or levels_are_divergent
    else:
        divergent = False
    if divergent:
        if vlim is None:
            vlim = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = (-vlim, vlim)
    vmin += center
    vmax += center
    if norm is not None:
        if norm.vmin is None:
            norm.vmin = vmin
        else:
            if not vmin_was_none and vmin != norm.vmin:
                raise ValueError('Cannot supply vmin and a norm with a different vmin.')
            vmin = norm.vmin
        if norm.vmax is None:
            norm.vmax = vmax
        else:
            if not vmax_was_none and vmax != norm.vmax:
                raise ValueError('Cannot supply vmax and a norm with a different vmax.')
            vmax = norm.vmax
    if isinstance(norm, mpl.colors.BoundaryNorm):
        levels = norm.boundaries
    if cmap is None:
        if divergent:
            cmap = OPTIONS['cmap_divergent']
        else:
            cmap = OPTIONS['cmap_sequential']
    if levels is not None:
        if is_scalar(levels):
            if user_minmax:
                levels = np.linspace(vmin, vmax, levels)
            elif levels == 1:
                levels = np.asarray([(vmin + vmax) / 2])
            else:
                ticker = mpl.ticker.MaxNLocator(levels - 1)
                levels = ticker.tick_values(vmin, vmax)
        vmin, vmax = (levels[0], levels[-1])
    if vmin == vmax:
        vmin, vmax = mpl.ticker.LinearLocator(2).tick_values(vmin, vmax)
    if extend is None:
        extend = _determine_extend(calc_data, vmin, vmax)
    if levels is not None and (not isinstance(norm, mpl.colors.BoundaryNorm)):
        cmap, newnorm = _build_discrete_cmap(cmap, levels, extend, filled)
        norm = newnorm if norm is None else norm
    if norm is not None:
        vmin = None
        vmax = None
    return dict(vmin=vmin, vmax=vmax, cmap=cmap, extend=extend, levels=levels, norm=norm)