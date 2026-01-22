import functools
import re
import warnings
from .._utils import set_module
import numpy.core.numeric as NX
from numpy.core import (isscalar, abs, finfo, atleast_1d, hstack, dot, array,
from numpy.core import overrides
from numpy.lib.twodim_base import diag, vander
from numpy.lib.function_base import trim_zeros
from numpy.lib.type_check import iscomplex, real, imag, mintypecode
from numpy.linalg import eigvals, lstsq, inv
def _raise_power(astr, wrap=70):
    n = 0
    line1 = ''
    line2 = ''
    output = ' '
    while True:
        mat = _poly_mat.search(astr, n)
        if mat is None:
            break
        span = mat.span()
        power = mat.groups()[0]
        partstr = astr[n:span[0]]
        n = span[1]
        toadd2 = partstr + ' ' * (len(power) - 1)
        toadd1 = ' ' * (len(partstr) - 1) + power
        if len(line2) + len(toadd2) > wrap or len(line1) + len(toadd1) > wrap:
            output += line1 + '\n' + line2 + '\n '
            line1 = toadd1
            line2 = toadd2
        else:
            line2 += partstr + ' ' * (len(power) - 1)
            line1 += ' ' * (len(partstr) - 1) + power
    output += line1 + '\n' + line2
    return output + astr[n:]