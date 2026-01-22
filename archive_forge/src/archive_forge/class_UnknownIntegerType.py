import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
class UnknownIntegerType(BasePrimitiveType):
    _attrs_ = ('name',)

    def __init__(self, name):
        self.name = name
        self.c_name_with_marker = name + '&'

    def is_integer_type(self):
        return True

    def build_backend_type(self, ffi, finishlist):
        raise NotImplementedError("integer type '%s' can only be used after compilation" % self.name)