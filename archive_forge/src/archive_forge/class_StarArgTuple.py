from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class StarArgTuple(_StarArgTupleMixin, Tuple):
    """To distinguish from Tuple() used as argument to a `*args`.
    """

    def __new__(cls, types):
        _HeterogeneousTuple.is_types_iterable(types)
        if types and all((t == types[0] for t in types[1:])):
            return StarArgUniTuple(dtype=types[0], count=len(types))
        else:
            return object.__new__(StarArgTuple)