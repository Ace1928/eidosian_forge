from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class Omitted(Opaque):
    """
    An omitted function argument with a default value.
    """

    def __init__(self, value):
        self._value = value
        self._value_key = get_hashable_key(value)
        super(Omitted, self).__init__('omitted(default=%r)' % (value,))

    @property
    def key(self):
        return (type(self._value), self._value_key)

    @property
    def value(self):
        return self._value