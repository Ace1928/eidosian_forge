import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _transform_pandas_df(data: DataFrame, enable_categorical: bool, feature_names: Optional[FeatureNames]=None, feature_types: Optional[FeatureTypes]=None, meta: Optional[str]=None, meta_type: Optional[NumpyDType]=None) -> Tuple[np.ndarray, Optional[FeatureNames], Optional[FeatureTypes]]:
    pyarrow_extension = False
    for dtype in data.dtypes:
        if not (dtype.name in _pandas_dtype_mapper or is_pd_sparse_dtype(dtype) or (is_pd_cat_dtype(dtype) and enable_categorical) or is_pa_ext_dtype(dtype)):
            _invalid_dataframe_dtype(data)
        if is_pa_ext_dtype(dtype):
            pyarrow_extension = True
    feature_names, feature_types = pandas_feature_info(data, meta, feature_names, feature_types, enable_categorical)
    transformed = pandas_cat_null(data)
    if pyarrow_extension:
        if transformed is data:
            transformed = data.copy(deep=False)
        transformed = pandas_ext_num_types(transformed)
    if meta and len(data.columns) > 1 and (meta not in _matrix_meta):
        raise ValueError(f'DataFrame for {meta} cannot have multiple columns')
    dtype = meta_type if meta_type else np.float32
    arr: np.ndarray = transformed.values
    if meta_type:
        arr = arr.astype(dtype)
    return (arr, feature_names, feature_types)