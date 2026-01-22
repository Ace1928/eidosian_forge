import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_typedesc(f):
    """Function to read in a type descriptor"""
    typedesc = {'typecode': _read_long(f), 'varflags': _read_long(f)}
    if typedesc['varflags'] & 2 == 2:
        raise Exception('System variables not implemented')
    typedesc['array'] = typedesc['varflags'] & 4 == 4
    typedesc['structure'] = typedesc['varflags'] & 32 == 32
    if typedesc['structure']:
        typedesc['array_desc'] = _read_arraydesc(f)
        typedesc['struct_desc'] = _read_structdesc(f)
    elif typedesc['array']:
        typedesc['array_desc'] = _read_arraydesc(f)
    return typedesc