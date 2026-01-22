import functools
import re
from .ctx_base import StandardBaseContext
from .libmp.backend import basestring, BACKEND
from . import libmp
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import function_docs
from . import rational
from .ctx_mp_python import _mpf, _mpc, mpnumeric
def _agm(ctx, a, b=1):
    prec, rounding = ctx._prec_rounding
    if hasattr(a, '_mpf_') and hasattr(b, '_mpf_'):
        try:
            v = libmp.mpf_agm(a._mpf_, b._mpf_, prec, rounding)
            return ctx.make_mpf(v)
        except ComplexResult:
            pass
    if hasattr(a, '_mpf_'):
        a = (a._mpf_, libmp.fzero)
    else:
        a = a._mpc_
    if hasattr(b, '_mpf_'):
        b = (b._mpf_, libmp.fzero)
    else:
        b = b._mpc_
    return ctx.make_mpc(libmp.mpc_agm(a, b, prec, rounding))