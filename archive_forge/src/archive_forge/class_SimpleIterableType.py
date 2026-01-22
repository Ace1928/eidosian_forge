from .abstract import ArrayCompatible, Dummy, IterableType, IteratorType
from numba.core.errors import NumbaTypeError, NumbaValueError
class SimpleIterableType(IterableType):

    def __init__(self, name, iterator_type):
        self._iterator_type = iterator_type
        super(SimpleIterableType, self).__init__(name)

    @property
    def iterator_type(self):
        return self._iterator_type