import os
import sys
from enum import Enum, _simple_enum
def _parse_mac(word):
    parts = word.split(_MAC_DELIM)
    if len(parts) != 6:
        return
    if _MAC_OMITS_LEADING_ZEROES:
        if not all((1 <= len(part) <= 2 for part in parts)):
            return
        hexstr = b''.join((part.rjust(2, b'0') for part in parts))
    else:
        if not all((len(part) == 2 for part in parts)):
            return
        hexstr = b''.join(parts)
    try:
        return int(hexstr, 16)
    except ValueError:
        return