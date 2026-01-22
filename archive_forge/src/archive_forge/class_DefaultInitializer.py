import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
class DefaultInitializer(InitializerBase):
    """Initializer wrapper that maps exceptions to default values.


    Parameters
    ----------
    initializer: :py:class`InitializerBase`
        the Initializer instance to wrap

    default:
        the value to return inlieu of the caught exception(s)

    exceptions: Exception or tuple
        the single Exception or tuple of Exceptions to catch and return
        the default value.

    """
    __slots__ = ('_initializer', '_default', '_exceptions')

    def __init__(self, initializer, default, exceptions):
        self._initializer = initializer
        self._default = default
        self._exceptions = exceptions

    def __call__(self, parent, index):
        try:
            return self._initializer(parent, index)
        except self._exceptions:
            return self._default

    def constant(self):
        """Return True if this initializer is constant across all indices"""
        return self._initializer.constant()

    def contains_indices(self):
        """Return True if this initializer contains embedded indices"""
        return self._initializer.contains_indices()

    def indices(self):
        return self._initializer.indices()