import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_tagdesc(f):
    """Function to read in a tag descriptor"""
    tagdesc = {'offset': _read_long(f)}
    if tagdesc['offset'] == -1:
        tagdesc['offset'] = _read_uint64(f)
    tagdesc['typecode'] = _read_long(f)
    tagflags = _read_long(f)
    tagdesc['array'] = tagflags & 4 == 4
    tagdesc['structure'] = tagflags & 32 == 32
    tagdesc['scalar'] = tagdesc['typecode'] in DTYPE_DICT
    return tagdesc