import sys, os
import textwrap
def _parse_num(val, type):
    if val[:2].lower() == '0x':
        radix = 16
    elif val[:2].lower() == '0b':
        radix = 2
        val = val[2:] or '0'
    elif val[:1] == '0':
        radix = 8
    else:
        radix = 10
    return type(val, radix)