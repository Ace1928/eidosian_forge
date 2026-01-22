import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.util import tf_export
def _result_type(*arrays_and_dtypes):
    """Returns the resulting type given a set of arrays."""

    def preprocess_float(x):
        if is_prefer_float32():
            if isinstance(x, float):
                return np.float32(x)
            elif isinstance(x, complex):
                return np.complex64(x)
        return x
    arrays_and_dtypes = [preprocess_float(x) for x in arrays_and_dtypes]
    dtype = np.result_type(*arrays_and_dtypes)
    return dtypes.as_dtype(canonicalize_dtype(dtype))