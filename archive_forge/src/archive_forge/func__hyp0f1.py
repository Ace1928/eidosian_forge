from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def _hyp0f1(ctx, b_s, z, **kwargs):
    (b, btype), = b_s
    if z:
        magz = ctx.mag(z)
    else:
        magz = 0
    if magz >= 8 and (not kwargs.get('force_series')):
        try:
            orig = ctx.prec
            try:
                ctx.prec += 12 + magz // 2

                def h():
                    w = ctx.sqrt(-z)
                    jw = ctx.j * w
                    u = 1 / (4 * jw)
                    c = ctx.mpq_1_2 - b
                    E = ctx.exp(2 * jw)
                    T1 = ([-jw, E], [c, -1], [], [], [b - ctx.mpq_1_2, ctx.mpq_3_2 - b], [], -u)
                    T2 = ([jw, E], [c, 1], [], [], [b - ctx.mpq_1_2, ctx.mpq_3_2 - b], [], u)
                    return (T1, T2)
                v = ctx.hypercomb(h, [], force_series=True)
                v = ctx.gamma(b) / (2 * ctx.sqrt(ctx.pi)) * v
            finally:
                ctx.prec = orig
            if ctx._is_real_type(b) and ctx._is_real_type(z):
                v = ctx._re(v)
            return +v
        except ctx.NoConvergence:
            pass
    return ctx.hypsum(0, 1, (btype,), [b], z, **kwargs)