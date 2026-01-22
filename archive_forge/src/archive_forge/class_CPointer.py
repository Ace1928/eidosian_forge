from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class CPointer(Type):
    """
    Type class for pointers to other types.

    Attributes
    ----------
        dtype : The pointee type
        addrspace : int
            The address space pointee belongs to.
    """
    mutable = True

    def __init__(self, dtype, addrspace=None):
        self.dtype = dtype
        self.addrspace = addrspace
        if addrspace is not None:
            name = '%s_%s*' % (dtype, addrspace)
        else:
            name = '%s*' % dtype
        super(CPointer, self).__init__(name)

    @property
    def key(self):
        return (self.dtype, self.addrspace)