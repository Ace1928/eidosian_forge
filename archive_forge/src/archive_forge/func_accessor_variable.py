import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def accessor_variable(name):
    key = 'variable ' + name
    tp, _ = ffi._parser._declarations[key]
    BType = ffi._get_cached_btype(tp)
    read_variable = backendlib.read_variable
    write_variable = backendlib.write_variable
    setattr(FFILibrary, name, property(lambda self: read_variable(BType, name), lambda self, value: write_variable(BType, name, value)))