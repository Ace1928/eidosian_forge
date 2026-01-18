import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def unknown_ptr_type(name, structname=None):
    if structname is None:
        structname = '$$%s' % name
    tp = StructType(structname, None, None, None)
    return NamedPointerType(tp, name)