import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
def _wrap_method_output(f, method):
    """Wrapper used by `_SetOutputMixin` to automatically wrap methods."""

    @wraps(f)
    def wrapped(self, X, *args, **kwargs):
        data_to_wrap = f(self, X, *args, **kwargs)
        if isinstance(data_to_wrap, tuple):
            return_tuple = (_wrap_data_with_container(method, data_to_wrap[0], X, self), *data_to_wrap[1:])
            if hasattr(type(data_to_wrap), '_make'):
                return type(data_to_wrap)._make(return_tuple)
            return return_tuple
        return _wrap_data_with_container(method, data_to_wrap, X, self)
    return wrapped