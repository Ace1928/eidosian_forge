import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def _wrap_mpi_function(ctx, f_real, f_complex=None):

    def g(x, **kwargs):
        if kwargs:
            prec = kwargs.get('prec', ctx._prec[0])
        else:
            prec = ctx._prec[0]
        x = ctx.convert(x)
        if hasattr(x, '_mpi_'):
            return ctx.make_mpf(f_real(x._mpi_, prec))
        if hasattr(x, '_mpci_'):
            return ctx.make_mpc(f_complex(x._mpci_, prec))
        raise ValueError
    return g