from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import is_supported_dtype
from pandas._typing import (
from pandas.compat import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import BaseMaskedDtype
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.algorithms import (
from pandas.core.array_algos import (
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import ExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import invalid_comparison
from pandas.core.util.hashing import hash_array
from pandas.compat.numpy import function as nv
def _wrap_na_result(self, *, name, axis, mask_size):
    mask = np.ones(mask_size, dtype=bool)
    float_dtyp = 'float32' if self.dtype == 'Float32' else 'float64'
    if name in ['mean', 'median', 'var', 'std', 'skew', 'kurt']:
        np_dtype = float_dtyp
    elif name in ['min', 'max'] or self.dtype.itemsize == 8:
        np_dtype = self.dtype.numpy_dtype.name
    else:
        is_windows_or_32bit = is_platform_windows() or not IS64
        int_dtyp = 'int32' if is_windows_or_32bit else 'int64'
        uint_dtyp = 'uint32' if is_windows_or_32bit else 'uint64'
        np_dtype = {'b': int_dtyp, 'i': int_dtyp, 'u': uint_dtyp, 'f': float_dtyp}[self.dtype.kind]
    value = np.array([1], dtype=np_dtype)
    return self._maybe_mask_result(value, mask=mask)