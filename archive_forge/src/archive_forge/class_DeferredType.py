from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class DeferredType(Type):
    """
    Represents a type that will be defined later.  It must be defined
    before it is materialized (used in the compiler).  Once defined, it
    behaves exactly as the type it is defining.
    """

    def __init__(self):
        self._define = None
        name = '{0}#{1}'.format(type(self).__name__, id(self))
        super(DeferredType, self).__init__(name)

    def get(self):
        if self._define is None:
            raise RuntimeError('deferred type not defined')
        return self._define

    def define(self, typ):
        if self._define is not None:
            raise TypeError('deferred type already defined')
        if not isinstance(typ, Type):
            raise TypeError('arg is not a Type; got: {0}'.format(type(typ)))
        self._define = typ

    def unify(self, typingctx, other):
        return typingctx.unify_pairs(self.get(), other)