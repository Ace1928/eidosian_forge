import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
class ScalarCallInitializer(InitializerBase):
    """Initializer for functions taking only the parent block argument."""
    __slots__ = ('_fcn', '_constant')

    def __init__(self, _fcn, constant=True):
        self._fcn = _fcn
        self._constant = constant

    def __call__(self, parent, idx):
        return self._fcn(parent)

    def constant(self):
        """Return True if this initializer is constant across all indices"""
        return self._constant