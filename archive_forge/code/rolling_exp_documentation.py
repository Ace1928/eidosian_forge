from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Generic
import numpy as np
from packaging.version import Version
from xarray.core.computation import apply_ufunc
from xarray.core.options import _get_keep_attrs
from xarray.core.pdcompat import count_not_none
from xarray.core.types import T_DataWithCoords
from xarray.core.utils import module_available
from xarray.namedarray import pycompat

        Exponentially weighted moving correlation.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").corr(da.shift(x=1))
        <xarray.DataArray (x: 5)> Size: 40B
        array([       nan,        nan,        nan, 0.4330127 , 0.48038446])
        Dimensions without coordinates: x
        