import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def _data_from_pandas(data: pd_DataFrame, feature_name: _LGBM_FeatureNameConfiguration, categorical_feature: _LGBM_CategoricalFeatureConfiguration, pandas_categorical: Optional[List[List]]) -> Tuple[np.ndarray, List[str], Union[List[str], List[int]], List[List]]:
    if len(data.shape) != 2 or data.shape[0] < 1:
        raise ValueError('Input data must be 2 dimensional and non empty.')
    data = data.copy(deep=False)
    if feature_name == 'auto':
        feature_name = [str(col) for col in data.columns]
    cat_cols = [col for col, dtype in zip(data.columns, data.dtypes) if isinstance(dtype, pd_CategoricalDtype)]
    cat_cols_not_ordered: List[str] = [col for col in cat_cols if not data[col].cat.ordered]
    if pandas_categorical is None:
        pandas_categorical = [list(data[col].cat.categories) for col in cat_cols]
    else:
        if len(cat_cols) != len(pandas_categorical):
            raise ValueError('train and valid dataset categorical_feature do not match.')
        for col, category in zip(cat_cols, pandas_categorical):
            if list(data[col].cat.categories) != list(category):
                data[col] = data[col].cat.set_categories(category)
    if len(cat_cols):
        data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
    if categorical_feature == 'auto':
        categorical_feature = cat_cols_not_ordered
    df_dtypes = [dtype.type for dtype in data.dtypes]
    df_dtypes.append(np.float32)
    target_dtype = np.result_type(*df_dtypes)
    return (_pandas_to_numpy(data, target_dtype=target_dtype), feature_name, categorical_feature, pandas_categorical)