import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_arraydesc(f):
    """Function to read in an array descriptor"""
    arraydesc = {'arrstart': _read_long(f)}
    if arraydesc['arrstart'] == 8:
        _skip_bytes(f, 4)
        arraydesc['nbytes'] = _read_long(f)
        arraydesc['nelements'] = _read_long(f)
        arraydesc['ndims'] = _read_long(f)
        _skip_bytes(f, 8)
        arraydesc['nmax'] = _read_long(f)
        arraydesc['dims'] = [_read_long(f) for _ in range(arraydesc['nmax'])]
    elif arraydesc['arrstart'] == 18:
        warnings.warn('Using experimental 64-bit array read', stacklevel=3)
        _skip_bytes(f, 8)
        arraydesc['nbytes'] = _read_uint64(f)
        arraydesc['nelements'] = _read_uint64(f)
        arraydesc['ndims'] = _read_long(f)
        _skip_bytes(f, 8)
        arraydesc['nmax'] = 8
        arraydesc['dims'] = []
        for d in range(arraydesc['nmax']):
            v = _read_long(f)
            if v != 0:
                raise Exception('Expected a zero in ARRAY_DESC')
            arraydesc['dims'].append(_read_long(f))
    else:
        raise Exception('Unknown ARRSTART: %i' % arraydesc['arrstart'])
    return arraydesc