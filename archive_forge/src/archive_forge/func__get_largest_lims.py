from __future__ import annotations
import functools
import itertools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast
import numpy as np
from xarray.core.formatting import format_item
from xarray.core.types import HueStyleOptions, T_DataArrayOrSet
from xarray.plot.utils import (
def _get_largest_lims(self) -> dict[str, tuple[float, float]]:
    """
        Get largest limits in the facetgrid.

        Returns
        -------
        lims_largest : dict[str, tuple[float, float]]
            Dictionary with the largest limits along each axis.

        Examples
        --------
        >>> ds = xr.tutorial.scatter_example_dataset(seed=42)
        >>> fg = ds.plot.scatter(x="A", y="B", hue="y", row="x", col="w")
        >>> round(fg._get_largest_lims()["x"][0], 3)
        -0.334
        """
    lims_largest: dict[str, tuple[float, float]] = dict(x=(np.inf, -np.inf), y=(np.inf, -np.inf), z=(np.inf, -np.inf))
    for axis in ('x', 'y', 'z'):
        lower, upper = lims_largest[axis]
        for ax in self.axs.flat:
            get_lim: None | Callable[[], tuple[float, float]] = getattr(ax, f'get_{axis}lim', None)
            if get_lim:
                lower_new, upper_new = get_lim()
                lower, upper = (min(lower, lower_new), max(upper, upper_new))
        lims_largest[axis] = (lower, upper)
    return lims_largest