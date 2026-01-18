import functools
import re
from .ctx_base import StandardBaseContext
from .libmp.backend import basestring, BACKEND
from . import libmp
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import function_docs
from . import rational
from .ctx_mp_python import _mpf, _mpc, mpnumeric
def workprec(ctx, n, normalize_output=False):
    """
        The block

            with workprec(n):
                <code>

        sets the precision to n bits, executes <code>, and then restores
        the precision.

        workprec(n)(f) returns a decorated version of the function f
        that sets the precision to n bits before execution,
        and restores the precision afterwards. With normalize_output=True,
        it rounds the return value to the parent precision.
        """
    return PrecisionManager(ctx, lambda p: n, None, normalize_output)