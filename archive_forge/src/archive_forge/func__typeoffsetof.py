import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _typeoffsetof(self, ctype, field_or_index, *fields_or_indexes):
    ctype, offset = self._backend.typeoffsetof(ctype, field_or_index)
    for field1 in fields_or_indexes:
        ctype, offset1 = self._backend.typeoffsetof(ctype, field1, 1)
        offset += offset1
    return (ctype, offset)