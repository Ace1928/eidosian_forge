from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def appellf3(ctx, a1, a2, b1, b2, c, x, y, **kwargs):
    outer_polynomial = ctx.isnpint(a1) or ctx.isnpint(b1)
    inner_polynomial = ctx.isnpint(a2) or ctx.isnpint(b2)
    if not outer_polynomial:
        if inner_polynomial or abs(x) > abs(y):
            x, y = (y, x)
            a1, a2, b1, b2 = (a2, a1, b2, b1)
    return ctx.hyper2d({'m': [a1, b1], 'n': [a2, b2]}, {'m+n': [c]}, x, y, **kwargs)