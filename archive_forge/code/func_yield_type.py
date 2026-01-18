from .abstract import ArrayCompatible, Dummy, IterableType, IteratorType
from numba.core.errors import NumbaTypeError, NumbaValueError
@property
def yield_type(self):
    return self._yield_type