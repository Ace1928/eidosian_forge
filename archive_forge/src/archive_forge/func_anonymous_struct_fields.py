import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def anonymous_struct_fields(self):
    if self.fldtypes is not None:
        for name, type in zip(self.fldnames, self.fldtypes):
            if name == '' and isinstance(type, StructOrUnion):
                yield type