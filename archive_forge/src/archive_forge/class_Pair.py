from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class Pair(Type):
    """
    A heterogeneous pair.
    """

    def __init__(self, first_type, second_type):
        self.first_type = first_type
        self.second_type = second_type
        name = 'pair<%s, %s>' % (first_type, second_type)
        super(Pair, self).__init__(name=name)

    @property
    def key(self):
        return (self.first_type, self.second_type)

    def unify(self, typingctx, other):
        if isinstance(other, Pair):
            first = typingctx.unify_pairs(self.first_type, other.first_type)
            second = typingctx.unify_pairs(self.second_type, other.second_type)
            if first is not None and second is not None:
                return Pair(first, second)