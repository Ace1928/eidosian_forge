import re
import struct
import binascii
def _input_type_check(s):
    try:
        m = memoryview(s)
    except TypeError as err:
        msg = 'expected bytes-like object, not %s' % s.__class__.__name__
        raise TypeError(msg) from err
    if m.format not in ('c', 'b', 'B'):
        msg = 'expected single byte elements, not %r from %s' % (m.format, s.__class__.__name__)
        raise TypeError(msg)
    if m.ndim != 1:
        msg = 'expected 1-D data, not %d-D data from %s' % (m.ndim, s.__class__.__name__)
        raise TypeError(msg)