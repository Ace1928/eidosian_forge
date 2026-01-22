import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def accessor_function(name):
    key = 'function ' + name
    tp, _ = ffi._parser._declarations[key]
    BType = ffi._get_cached_btype(tp)
    value = backendlib.load_function(BType, name)
    library.__dict__[name] = value