import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def _mpi(ctx, a, b=None):
    if b is None:
        return ctx.mpf(a)
    return ctx.mpf((a, b))