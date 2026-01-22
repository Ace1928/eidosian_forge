from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _add_constants(self, key, val):
    if key in self._int_constants:
        if self._int_constants[key] == val:
            return
        raise FFIError('multiple declarations of constant: %s' % (key,))
    self._int_constants[key] = val