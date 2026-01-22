from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def acot(ctx, z):
    if not z:
        return ctx.pi * 0.5
    else:
        return ctx.atan(ctx.one / z)