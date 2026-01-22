import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import (
def _infer_numpy_dtype(dtype) -> DataType:
    supported_types = np.dtype
    try:
        from pandas.core.dtypes.base import ExtensionDtype
        supported_types = (np.dtype, ExtensionDtype)
    except ImportError:
        pass
    if not isinstance(dtype, supported_types):
        raise TypeError(f"Expected numpy.dtype or pandas.ExtensionDtype, got '{type(dtype)}'.")
    if dtype.kind == 'b':
        return DataType.boolean
    elif dtype.kind == 'i' or dtype.kind == 'u':
        if dtype.itemsize < 4 or (dtype.kind == 'i' and dtype.itemsize == 4):
            return DataType.integer
        elif dtype.itemsize < 8 or (dtype.kind == 'i' and dtype.itemsize == 8):
            return DataType.long
    elif dtype.kind == 'f':
        if dtype.itemsize <= 4:
            return DataType.float
        elif dtype.itemsize <= 8:
            return DataType.double
    elif dtype.kind == 'U':
        return DataType.string
    elif dtype.kind == 'S':
        return DataType.binary
    elif dtype.kind == 'O':
        raise Exception('Can not infer object without looking at the values, call _map_numpy_array instead.')
    elif dtype.kind == 'M':
        return DataType.datetime
    raise MlflowException(f"Unsupported numpy data type '{dtype}', kind '{dtype.kind}'")