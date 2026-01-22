from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload
from xarray.core.alignment import broadcast
from xarray.plot import dataarray_plot
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
def _normalize_args(plotmethod: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    from xarray.core.dataarray import DataArray
    locals_ = dict(inspect.signature(getattr(DataArray().plot, plotmethod)).bind(*args, **kwargs).arguments.items())
    locals_.update(locals_.pop('kwargs', {}))
    return locals_