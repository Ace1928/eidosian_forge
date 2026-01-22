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
@property
def _values_unique(self) -> np.ndarray | None:
    """
        Return unique values.

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a)._values_unique
        array([1, 3, 5])

        >>> _Normalize(a, width=(18, 36, 72))._values_unique
        array([18., 45., 72.])

        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> _Normalize(a)._values_unique
        array([0. , 0.5, 2. , 3. ])

        >>> _Normalize(a, width=(18, 36, 72))._values_unique
        array([18., 27., 54., 72.])
        """
    if self.data is None:
        return None
    val: np.ndarray
    if self.data_is_numeric:
        val = self._data_unique
    else:
        val = self._indexes_centered(self._data_unique_index)
    return self._calc_widths(val)