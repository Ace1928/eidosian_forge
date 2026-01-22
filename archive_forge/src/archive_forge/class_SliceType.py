from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class SliceType(Type):

    def __init__(self, name, members):
        assert members in (2, 3)
        self.members = members
        self.has_step = members >= 3
        super(SliceType, self).__init__(name)

    @property
    def key(self):
        return self.members