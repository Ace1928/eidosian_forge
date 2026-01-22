from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def acsc(ctx, z):
    return ctx.asin(ctx.one / z)