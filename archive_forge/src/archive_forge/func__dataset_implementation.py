from __future__ import annotations
import functools
import itertools
import math
import warnings
from collections.abc import Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
import numpy as np
from packaging.version import Version
from xarray.core import dtypes, duck_array_ops, utils
from xarray.core.arithmetic import CoarsenArithmetic
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import CoarsenBoundaryOptions, SideOptions, T_Xarray
from xarray.core.utils import (
from xarray.namedarray import pycompat
def _dataset_implementation(self, func, keep_attrs, **kwargs):
    from xarray.core.dataset import Dataset
    keep_attrs = self._get_keep_attrs(keep_attrs)
    reduced = {}
    for key, da in self.obj.data_vars.items():
        if any((d in da.dims for d in self.dim)):
            reduced[key] = func(self.rollings[key], keep_attrs=keep_attrs, **kwargs)
        else:
            reduced[key] = self.obj[key].copy()
            if not keep_attrs:
                reduced[key].attrs = {}
    attrs = self.obj.attrs if keep_attrs else {}
    return Dataset(reduced, coords=self.obj.coords, attrs=attrs)