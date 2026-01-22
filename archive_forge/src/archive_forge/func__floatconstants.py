from __future__ import absolute_import
import re
import sys
import struct
from .compat import PY3, unichr
from .scanner import make_scanner, JSONDecodeError
def _floatconstants():
    if sys.version_info < (2, 6):
        _BYTES = '7FF80000000000007FF0000000000000'.decode('hex')
        nan, inf = struct.unpack('>dd', _BYTES)
    else:
        nan = float('nan')
        inf = float('inf')
    return (nan, inf, -inf)