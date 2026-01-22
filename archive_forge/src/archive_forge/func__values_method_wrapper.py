from __future__ import annotations
import operator
import numpy as np
from xarray.core import dtypes, duck_array_ops
def _values_method_wrapper(name):

    def func(self, *args, **kwargs):
        return _call_possibly_missing_method(self.data, name, args, kwargs)
    func.__name__ = name
    func.__doc__ = getattr(np.ndarray, name).__doc__
    return func