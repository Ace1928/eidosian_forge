import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def addressof(self, cdata, *fields_or_indexes):
    """Return the address of a <cdata 'struct-or-union'>.
        If 'fields_or_indexes' are given, returns the address of that
        field or array item in the structure or array, recursively in
        case of nested structures.
        """
    try:
        ctype = self._backend.typeof(cdata)
    except TypeError:
        if '__addressof__' in type(cdata).__dict__:
            return type(cdata).__addressof__(cdata, *fields_or_indexes)
        raise
    if fields_or_indexes:
        ctype, offset = self._typeoffsetof(ctype, *fields_or_indexes)
    else:
        if ctype.kind == 'pointer':
            raise TypeError('addressof(pointer)')
        offset = 0
    ctypeptr = self._pointer_to(ctype)
    return self._backend.rawaddressof(ctypeptr, cdata, offset)