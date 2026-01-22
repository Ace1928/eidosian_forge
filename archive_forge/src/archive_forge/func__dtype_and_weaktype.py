import functools
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.variables import ALLOWED_DTYPES
from keras.src.backend.common.variables import standardize_dtype
def _dtype_and_weaktype(value):
    """Return a (dtype, weak_type) tuple for the given input."""
    is_weak_type = False
    if value is int or value is float:
        is_weak_type = True
    return (standardize_dtype(value), is_weak_type)