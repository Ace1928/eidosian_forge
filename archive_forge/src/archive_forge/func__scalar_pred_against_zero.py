import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def _scalar_pred_against_zero(builder, value, fpred, icond):
    nullval = value.type(0)
    if isinstance(value.type, (ir.FloatType, ir.DoubleType)):
        isnull = fpred(value, nullval)
    elif isinstance(value.type, ir.IntType):
        isnull = builder.icmp_signed(icond, value, nullval)
    else:
        raise TypeError('unexpected value type %s' % (value.type,))
    return isnull