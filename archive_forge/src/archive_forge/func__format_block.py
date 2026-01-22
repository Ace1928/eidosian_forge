import os
import numpy as np
from copy import deepcopy
from ase.calculators.calculator import KPoints, kpts2kpts
def _format_block(key, val, nindent=0):
    prefix = '  ' * nindent
    prefix2 = '  ' * (nindent + 1)
    if val is None:
        return [prefix + key]
    if not isinstance(val, dict):
        return [prefix + _format_line(key, val)]
    out = [prefix + key]
    for subkey, subval in val.items():
        if (key, subkey) in _special_keypairs:
            if (key, subkey) == ('nwpw', 'brillouin_zone'):
                out += _format_brillouin_zone(subval)
            else:
                out += _format_block(subkey, subval, nindent + 1)
        else:
            if isinstance(subval, dict):
                subval = ' '.join([_format_line(a, b) for a, b in subval.items()])
            out.append(prefix2 + ' '.join([_format_line(subkey, subval)]))
    out.append(prefix + 'end')
    return out