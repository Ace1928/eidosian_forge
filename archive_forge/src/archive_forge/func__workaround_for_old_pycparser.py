from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _workaround_for_old_pycparser(csource):
    parts = []
    while True:
        match = _r_star_const_space.search(csource)
        if not match:
            break
        parts.append(csource[:match.start()])
        parts.append('(')
        closing = ')'
        parts.append(match.group())
        endpos = match.end()
        if csource.startswith('*', endpos):
            parts.append('(')
            closing += ')'
        level = 0
        i = endpos
        while i < len(csource):
            c = csource[i]
            if c == '(':
                level += 1
            elif c == ')':
                if level == 0:
                    break
                level -= 1
            elif c in ',;=':
                if level == 0:
                    break
            i += 1
        csource = csource[endpos:i] + closing + csource[i:]
    parts.append(csource)
    return ''.join(parts)