import numbers
import os
from packaging.version import Version
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._typing import Dtype
from pandas.compat import set_function_name
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.indexers import check_array_indexer, validate_indices
from pandas.io.formats.format import ExtensionArrayFormatter
from ray.air.util.tensor_extensions.utils import (
from ray.util.annotations import PublicAPI
@classmethod
def _create_method(cls, op, coerce_to_dtype=True, result_dtype=None):
    """
        Add support for binary operators by unwrapping, applying, and
        rewrapping.
        """

    def _binop(self, other):
        lvalues = self._tensor
        if isinstance(other, (ABCDataFrame, ABCSeries, ABCIndex)):
            return NotImplemented
        if op_name in ['__divmod__', '__rdivmod__']:
            raise NotImplementedError
        if isinstance(other, (TensorArray, TensorArrayElement)):
            rvalues = other._tensor
        else:
            rvalues = other
        result = op(lvalues, rvalues)
        if isinstance(self, TensorArrayElement) and (not isinstance(other, TensorArrayElement) or not np.isscalar(other)):
            result_wrapped = TensorArray(result)
        else:
            result_wrapped = cls(result)
        return result_wrapped
    op_name = f'__{op.__name__}__'
    return set_function_name(_binop, op_name, cls)