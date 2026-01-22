from __future__ import absolute_import
import functools
import json
import re
import sys
@staticmethod
def __utf16_decode_surrogate_pair(leading, trailing):
    """Returns the unicode code point corresponding to leading surrogate
        'leading' and trailing surrogate 'trailing'.  The return value will not
        make any sense if 'leading' or 'trailing' are not in the correct ranges
        for leading or trailing surrogates."""
    w = leading >> 6 & 15
    u = w + 1
    x0 = leading & 63
    x1 = trailing & 1023
    return u << 16 | x0 << 10 | x1