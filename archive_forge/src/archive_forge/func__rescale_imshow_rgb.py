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
def _rescale_imshow_rgb(darray, vmin, vmax, robust):
    assert robust or vmin is not None or vmax is not None
    if robust:
        if vmax is None:
            vmax = np.nanpercentile(darray, 100 - ROBUST_PERCENTILE)
        if vmin is None:
            vmin = np.nanpercentile(darray, ROBUST_PERCENTILE)
    elif vmax is None:
        vmax = 255 if np.issubdtype(darray.dtype, np.integer) else 1
        if vmax < vmin:
            raise ValueError(f'vmin={vmin!r} is less than the default vmax ({vmax!r}) - you must supply a vmax > vmin in this case.')
    elif vmin is None:
        vmin = 0
        if vmin > vmax:
            raise ValueError(f'vmax={vmax!r} is less than the default vmin (0) - you must supply a vmin < vmax in this case.')
    darray = ((darray.astype('f8') - vmin) / (vmax - vmin)).astype('f4')
    return np.minimum(np.maximum(darray, 0), 1)