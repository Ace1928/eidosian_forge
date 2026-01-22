import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _proxy_transform(data: DataType, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes], enable_categorical: bool) -> TransformedData:
    if _is_cudf_df(data) or _is_cudf_ser(data):
        return _transform_cudf_df(data, feature_names, feature_types, enable_categorical)
    if _is_cupy_array(data):
        data = _transform_cupy_array(data)
        return (data, None, feature_names, feature_types)
    if _is_dlpack(data):
        return (_transform_dlpack(data), None, feature_names, feature_types)
    if _is_list(data) or _is_tuple(data):
        data = np.array(data)
    if _is_np_array_like(data):
        data, _ = _ensure_np_dtype(data, data.dtype)
        return (data, None, feature_names, feature_types)
    if _is_scipy_csr(data):
        data = transform_scipy_sparse(data, True)
        return (data, None, feature_names, feature_types)
    if _is_pandas_series(data):
        import pandas as pd
        data = pd.DataFrame(data)
    if _is_pandas_df(data):
        arr, feature_names, feature_types = _transform_pandas_df(data, enable_categorical, feature_names, feature_types)
        arr, _ = _ensure_np_dtype(arr, arr.dtype)
        return (arr, None, feature_names, feature_types)
    raise TypeError('Value type is not supported for data iterator:' + str(type(data)))