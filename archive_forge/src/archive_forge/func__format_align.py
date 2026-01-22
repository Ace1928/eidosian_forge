import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _format_align(sign, body, spec):
    """Given an unpadded, non-aligned numeric string 'body' and sign
    string 'sign', add padding and alignment conforming to the given
    format specifier dictionary 'spec' (as produced by
    parse_format_specifier).

    """
    minimumwidth = spec['minimumwidth']
    fill = spec['fill']
    padding = fill * (minimumwidth - len(sign) - len(body))
    align = spec['align']
    if align == '<':
        result = sign + body + padding
    elif align == '>':
        result = padding + sign + body
    elif align == '=':
        result = sign + padding + body
    elif align == '^':
        half = len(padding) // 2
        result = padding[:half] + sign + body + padding[half:]
    else:
        raise ValueError('Unrecognised alignment field')
    return result