from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def _hyp1f1(ctx, a_s, b_s, z, **kwargs):
    (a, atype), = a_s
    (b, btype), = b_s
    if not z:
        return ctx.one + z
    magz = ctx.mag(z)
    if magz >= 7 and (not (ctx.isint(a) and ctx.re(a) <= 0)):
        if ctx.isinf(z):
            if ctx.sign(a) == ctx.sign(b) == ctx.sign(z) == 1:
                return ctx.inf
            return ctx.nan * z
        try:
            try:
                ctx.prec += magz
                sector = ctx._im(z) < 0

                def h(a, b):
                    if sector:
                        E = ctx.expjpi(ctx.fneg(a, exact=True))
                    else:
                        E = ctx.expjpi(a)
                    rz = 1 / z
                    T1 = ([E, z], [1, -a], [b], [b - a], [a, 1 + a - b], [], -rz)
                    T2 = ([ctx.exp(z), z], [1, a - b], [b], [a], [b - a, 1 - a], [], rz)
                    return (T1, T2)
                v = ctx.hypercomb(h, [a, b], force_series=True)
                if ctx._is_real_type(a) and ctx._is_real_type(b) and ctx._is_real_type(z):
                    v = ctx._re(v)
                return +v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec -= magz
    v = ctx.hypsum(1, 1, (atype, btype), [a, b], z, **kwargs)
    return v