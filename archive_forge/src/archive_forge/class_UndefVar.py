from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class UndefVar(Dummy):
    """
    A type that is created by Expr.undef to represent an undefined variable.
    This type can be promoted to any other type.
    This is introduced to handle Python 3.12 LOAD_FAST_AND_CLEAR.
    """

    def can_convert_to(self, typingctx, other):
        return Conversion.promote