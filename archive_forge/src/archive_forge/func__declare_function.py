from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _declare_function(self, tp, quals, decl):
    tp = self._get_type_pointer(tp, quals)
    if self._options.get('dllexport'):
        tag = 'dllexport_python '
    elif self._inside_extern_python == '__cffi_extern_python_start':
        tag = 'extern_python '
    elif self._inside_extern_python == '__cffi_extern_python_plus_c_start':
        tag = 'extern_python_plus_c '
    else:
        tag = 'function '
    self._declare(tag + decl.name, tp)