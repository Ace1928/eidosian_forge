import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
class InitializerBase(object):
    """Base class for all Initializer objects"""
    __slots__ = ()
    verified = False

    def __getstate__(self):
        """Class serializer

        This class must declare __getstate__ because it is slotized.
        This implementation should be sufficient for simple derived
        classes (where __slots__ are only declared on the most derived
        class).
        """
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state):
        for key, val in state.items():
            object.__setattr__(self, key, val)

    def constant(self):
        """Return True if this initializer is constant across all indices"""
        return False

    def contains_indices(self):
        """Return True if this initializer contains embedded indices"""
        return False

    def indices(self):
        """Return a generator over the embedded indices

        This will raise a RuntimeError if this initializer does not
        contain embedded indices
        """
        raise RuntimeError('Initializer %s does not contain embedded indices' % (type(self).__name__,))