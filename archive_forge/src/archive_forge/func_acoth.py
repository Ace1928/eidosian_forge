from ..libmp.backend import xrange
import math
import cmath
@defun_wrapped
def acoth(ctx, z):
    if not z:
        return ctx.pi * 0.5j
    else:
        return ctx.atanh(ctx.one / z)