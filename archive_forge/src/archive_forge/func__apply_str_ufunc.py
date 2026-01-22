from __future__ import annotations
import codecs
import re
import textwrap
from collections.abc import Hashable, Mapping
from functools import reduce
from operator import or_ as set_union
from re import Pattern
from typing import TYPE_CHECKING, Any, Callable, Generic
from unicodedata import normalize
import numpy as np
from xarray.core import duck_array_ops
from xarray.core.computation import apply_ufunc
from xarray.core.types import T_DataArray
def _apply_str_ufunc(*, func: Callable, obj: Any, dtype: DTypeLike=None, output_core_dims: list | tuple=((),), output_sizes: Mapping[Any, int] | None=None, func_args: tuple=(), func_kwargs: Mapping={}) -> Any:
    if dtype is None:
        dtype = obj.dtype
    dask_gufunc_kwargs = dict()
    if output_sizes is not None:
        dask_gufunc_kwargs['output_sizes'] = output_sizes
    return apply_ufunc(func, obj, *func_args, vectorize=True, dask='parallelized', output_dtypes=[dtype], output_core_dims=output_core_dims, dask_gufunc_kwargs=dask_gufunc_kwargs, **func_kwargs)