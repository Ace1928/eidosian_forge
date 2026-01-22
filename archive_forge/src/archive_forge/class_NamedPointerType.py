import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
class NamedPointerType(PointerType):
    _attrs_ = ('totype', 'name')

    def __init__(self, totype, name, quals=0):
        PointerType.__init__(self, totype, quals)
        self.name = name
        self.c_name_with_marker = name + '&'