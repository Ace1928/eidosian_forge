from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def _hyp2f2(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), (a2, a2type) = a_s
    (b1, b1type), (b2, b2type) = b_s
    absz = abs(z)
    magz = ctx.mag(z)
    orig = ctx.prec
    asymp_extraprec = magz
    can_use_asymptotic = not kwargs.get('force_series') and ctx.mag(absz) > 3
    if can_use_asymptotic:
        try:
            try:
                ctx.prec += asymp_extraprec

                def h(a1, a2, b1, b2):
                    X = a1 + a2 - b1 - b2
                    A2 = a1 + a2
                    B2 = b1 + b2
                    c = {}
                    c[0] = ctx.one
                    c[1] = (A2 - 1) * X + b1 * b2 - a1 * a2
                    s1 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = 1 - B2 + 2 * a1 + a1 ** 2 + 2 * a2 + a2 ** 2 - A2 * B2 + a1 * a2 + b1 * b2 + (2 * B2 - 3 * (A2 + 1)) * k + 2 * k ** 2
                            uu2 = (k - A2 + b1 - 1) * (k - A2 + b2 - 1) * (k - X - 2)
                            c[k] = ctx.one / k * (uu1 * c[k - 1] - uu2 * c[k - 2])
                        t1 = c[k] * z ** (-k)
                        if abs(t1) < 0.1 * ctx.eps:
                            break
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            raise ctx.NoConvergence
                        s1 += t1
                        tprev = t1
                        k += 1
                    S = ctx.exp(z) * s1
                    T1 = ([z, S], [X, 1], [b1, b2], [a1, a2], [], [], 0)
                    T2 = ([-z], [-a1], [b1, b2, a2 - a1], [a2, b1 - a1, b2 - a1], [a1, a1 - b1 + 1, a1 - b2 + 1], [a1 - a2 + 1], -1 / z)
                    T3 = ([-z], [-a2], [b1, b2, a1 - a2], [a1, b1 - a2, b2 - a2], [a2, a2 - b1 + 1, a2 - b2 + 1], [-a1 + a2 + 1], -1 / z)
                    return (T1, T2, T3)
                v = ctx.hypercomb(h, [a1, a2, b1, b2], force_series=True, maxterms=4 * ctx.prec)
                if sum((ctx._is_real_type(u) for u in [a1, a2, b1, b2, z])) == 5:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig
    return ctx.hypsum(2, 2, (a1type, a2type, b1type, b2type), [a1, a2, b1, b2], z, **kwargs)