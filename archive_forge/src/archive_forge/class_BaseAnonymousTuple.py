from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class BaseAnonymousTuple(BaseTuple):
    """
    Mixin for non-named tuples.
    """

    def can_convert_to(self, typingctx, other):
        """
        Convert this tuple to another one.  Note named tuples are rejected.
        """
        if not isinstance(other, BaseAnonymousTuple):
            return
        if len(self) != len(other):
            return
        if len(self) == 0:
            return Conversion.safe
        if isinstance(other, BaseTuple):
            kinds = [typingctx.can_convert(ta, tb) for ta, tb in zip(self, other)]
            if any((kind is None for kind in kinds)):
                return
            return max(kinds)

    def __unliteral__(self):
        return type(self).from_types([unliteral(t) for t in self])