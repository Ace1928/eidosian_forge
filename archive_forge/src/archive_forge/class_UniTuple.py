from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class UniTuple(BaseAnonymousTuple, _HomogeneousTuple, Sequence):
    """
    Type class for homogeneous tuples.
    """

    def __init__(self, dtype, count):
        self.dtype = dtype
        self.count = count
        name = '%s(%s x %d)' % (self.__class__.__name__, dtype, count)
        super(UniTuple, self).__init__(name)

    @property
    def mangling_args(self):
        return (self.__class__.__name__, (self.dtype, self.count))

    @property
    def key(self):
        return (self.dtype, self.count)

    def unify(self, typingctx, other):
        """
        Unify UniTuples with their dtype
        """
        if isinstance(other, UniTuple) and len(self) == len(other):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            if dtype is not None:
                return UniTuple(dtype=dtype, count=self.count)

    def __unliteral__(self):
        return type(self)(dtype=unliteral(self.dtype), count=self.count)

    def __repr__(self):
        return f'UniTuple({repr(self.dtype)}, {self.count})'