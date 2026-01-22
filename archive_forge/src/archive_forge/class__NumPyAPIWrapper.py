import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
class _NumPyAPIWrapper:
    """Array API compat wrapper for any numpy version

    NumPy < 1.22 does not expose the numpy.array_api namespace. This
    wrapper makes it possible to write code that uses the standard
    Array API while working with any version of NumPy supported by
    scikit-learn.

    See the `get_namespace()` public function for more details.
    """
    _CREATION_FUNCS = {'arange', 'empty', 'empty_like', 'eye', 'full', 'full_like', 'linspace', 'ones', 'ones_like', 'zeros', 'zeros_like'}
    _DTYPES = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64', 'complex64', 'complex128'}

    def __getattr__(self, name):
        attr = getattr(numpy, name)
        if name in self._CREATION_FUNCS:
            return _accept_device_cpu(attr)
        if name in self._DTYPES:
            return numpy.dtype(attr)
        return attr

    @property
    def bool(self):
        return numpy.bool_

    def astype(self, x, dtype, *, copy=True, casting='unsafe'):
        return x.astype(dtype, copy=copy, casting=casting)

    def asarray(self, x, *, dtype=None, device=None, copy=None):
        _check_device_cpu(device)
        if copy is True:
            return numpy.array(x, copy=True, dtype=dtype)
        else:
            return numpy.asarray(x, dtype=dtype)

    def unique_inverse(self, x):
        return numpy.unique(x, return_inverse=True)

    def unique_counts(self, x):
        return numpy.unique(x, return_counts=True)

    def unique_values(self, x):
        return numpy.unique(x)

    def concat(self, arrays, *, axis=None):
        return numpy.concatenate(arrays, axis=axis)

    def reshape(self, x, shape, *, copy=None):
        """Gives a new shape to an array without changing its data.

        The Array API specification requires shape to be a tuple.
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html
        """
        if not isinstance(shape, tuple):
            raise TypeError(f'shape must be a tuple, got {shape!r} of type {type(shape)}')
        if copy is True:
            x = x.copy()
        return numpy.reshape(x, shape)

    def isdtype(self, dtype, kind):
        return isdtype(dtype, kind, xp=self)