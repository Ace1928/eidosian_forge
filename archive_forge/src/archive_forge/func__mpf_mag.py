from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
def _mpf_mag(ctx, x):
    sign, man, exp, bc = x
    if man:
        return exp + bc
    if x == fzero:
        return ctx.ninf
    if x == finf or x == fninf:
        return ctx.inf
    return ctx.nan