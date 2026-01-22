import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def addressof_var(name):
    try:
        return addr_variables[name]
    except KeyError:
        with ffi._lock:
            if name not in addr_variables:
                key = 'variable ' + name
                tp, _ = ffi._parser._declarations[key]
                BType = ffi._get_cached_btype(tp)
                if BType.kind != 'array':
                    BType = model.pointer_cache(ffi, BType)
                p = backendlib.load_function(BType, name)
                addr_variables[name] = p
        return addr_variables[name]