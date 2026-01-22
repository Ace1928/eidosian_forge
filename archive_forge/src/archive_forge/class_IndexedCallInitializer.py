import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
class IndexedCallInitializer(InitializerBase):
    """Initializer for functions and callable objects"""
    __slots__ = ('_fcn',)

    def __init__(self, _fcn):
        self._fcn = _fcn

    def __call__(self, parent, idx):
        if idx.__class__ is tuple:
            return self._fcn(parent, *idx)
        else:
            return self._fcn(parent, idx)