import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _unpack_opargs(code):
    extended_arg = 0
    caches = 0
    for i in range(0, len(code), 2):
        if caches:
            caches -= 1
            continue
        op = code[i]
        deop = _deoptop(op)
        caches = _inline_cache_entries[deop]
        if deop >= HAVE_ARGUMENT:
            arg = code[i + 1] | extended_arg
            extended_arg = arg << 8 if deop == EXTENDED_ARG else 0
            if extended_arg >= _INT_OVERFLOW:
                extended_arg -= 2 * _INT_OVERFLOW
        else:
            arg = None
            extended_arg = 0
        yield (i, op, arg)