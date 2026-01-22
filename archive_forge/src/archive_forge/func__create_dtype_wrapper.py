from typing import Generic, TypeVar
from tensorflow.python.framework import dtypes as _dtypes
def _create_dtype_wrapper(name, underlying_dtype: _dtypes.DType):
    return type(name, (DTypeAnnotation,), {'underlying_dtype': underlying_dtype})