from __future__ import annotations
import functools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast, overload
import numpy as np
import pandas as pd
from xarray.core.alignment import broadcast
from xarray.core.concat import concat
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
def _prepare_plot1d_data(darray: T_DataArray, coords_to_plot: MutableMapping[str, Hashable], plotfunc_name: str | None=None, _is_facetgrid: bool=False) -> dict[str, T_DataArray]:
    """
    Prepare data for usage with plt.scatter.

    Parameters
    ----------
    darray : T_DataArray
        Base DataArray.
    coords_to_plot : MutableMapping[str, Hashable]
        Coords that will be plotted.
    plotfunc_name : str | None
        Name of the plotting function that will be used.

    Returns
    -------
    plts : dict[str, T_DataArray]
        Dict of DataArrays that will be sent to matplotlib.

    Examples
    --------
    >>> # Make sure int coords are plotted:
    >>> a = xr.DataArray(
    ...     data=[1, 2],
    ...     coords={1: ("x", [0, 1], {"units": "s"})},
    ...     dims=("x",),
    ...     name="a",
    ... )
    >>> plts = xr.plot.dataarray_plot._prepare_plot1d_data(
    ...     a, coords_to_plot={"x": 1, "z": None, "hue": None, "size": None}
    ... )
    >>> # Check which coords to plot:
    >>> print({k: v.name for k, v in plts.items()})
    {'y': 'a', 'x': 1}
    """
    if darray.ndim > 1:
        dims_T = []
        if np.issubdtype(darray.dtype, np.floating):
            for v in ['z', 'x']:
                dim = coords_to_plot.get(v, None)
                if dim is not None and dim in darray.dims:
                    darray_nan = np.nan * darray.isel({dim: -1})
                    darray = concat([darray, darray_nan], dim=dim)
                    dims_T.append(coords_to_plot[v])
        darray = darray.transpose(..., *dims_T)
        darray = darray.stack(_stacked_dim=darray.dims)
    plts = dict(y=darray)
    plts.update({k: darray.coords[v] for k, v in coords_to_plot.items() if v is not None})
    plts = dict(zip(plts.keys(), broadcast(*plts.values())))
    return plts