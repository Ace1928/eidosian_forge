from __future__ import annotations
import numbers
import numpy as np
from xarray.core._typed_ops import (
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.ops import IncludeNumpySameMethods, IncludeReduceMethods
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.namedarray.utils import is_duck_array
class DatasetArithmetic(ImplementsDatasetReduce, SupportsArithmetic, DatasetOpsMixin):
    __slots__ = ()
    __array_priority__ = 50