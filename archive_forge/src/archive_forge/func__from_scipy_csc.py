import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _from_scipy_csc(data: DataType, missing: FloatCompatible, nthread: int, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes]) -> DispatchedDataBackendReturnType:
    """Initialize data from a CSC matrix."""
    handle = ctypes.c_void_p()
    transform_scipy_sparse(data, False)
    _check_call(_LIB.XGDMatrixCreateFromCSC(_array_interface(data.indptr), _array_interface(data.indices), _array_interface(data.data), c_bst_ulong(data.shape[0]), make_jcargs(missing=float(missing), nthread=int(nthread)), ctypes.byref(handle)))
    return (handle, feature_names, feature_types)