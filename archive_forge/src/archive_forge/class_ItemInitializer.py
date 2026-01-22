import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
class ItemInitializer(InitializerBase):
    """Initializer for dict-like values supporting __getitem__()"""
    __slots__ = ('_dict',)

    def __init__(self, _dict):
        self._dict = _dict

    def __call__(self, parent, idx):
        return self._dict[idx]

    def contains_indices(self):
        return True

    def indices(self):
        try:
            return self._dict.keys()
        except AttributeError:
            return range(len(self._dict))