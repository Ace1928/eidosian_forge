from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def asech(ctx, z):
    return ctx.acosh(ctx.one / z)