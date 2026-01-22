from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def _hyp1f2(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), = a_s
    (b1, b1type), (b2, b2type) = b_s
    absz = abs(z)
    magz = ctx.mag(z)
    orig = ctx.prec
    asymp_extraprec = z and magz // 2
    can_use_asymptotic = not kwargs.get('force_series') and ctx.mag(absz) > 19 and (ctx.sqrt(absz) > 1.5 * orig)
    if can_use_asymptotic:
        try:
            try:
                ctx.prec += asymp_extraprec

                def h(a1, b1, b2):
                    X = ctx.mpq_1_2 * (a1 - b1 - b2 + ctx.mpq_1_2)
                    c = {}
                    c[0] = ctx.one
                    c[1] = 2 * (ctx.mpq_1_4 * (3 * a1 + b1 + b2 - 2) * (a1 - b1 - b2) + b1 * b2 - ctx.mpq_3_16)
                    c[2] = 2 * (b1 * b2 + ctx.mpq_1_4 * (a1 - b1 - b2) * (3 * a1 + b1 + b2 - 2) - ctx.mpq_3_16) ** 2 + ctx.mpq_1_16 * (-16 * (2 * a1 - 3) * b1 * b2 + 4 * (a1 - b1 - b2) * (-8 * a1 ** 2 + 11 * a1 + b1 + b2 - 2) - 3)
                    s1 = 0
                    s2 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = 3 * k ** 2 + (-6 * a1 + 2 * b1 + 2 * b2 - 4) * k + 3 * a1 ** 2 - (b1 - b2) ** 2 - 2 * a1 * (b1 + b2 - 2) + ctx.mpq_1_4
                            uu2 = (k - a1 + b1 - b2 - ctx.mpq_1_2) * (k - a1 - b1 + b2 - ctx.mpq_1_2) * (k - a1 + b1 + b2 - ctx.mpq_5_2)
                            c[k] = ctx.one / (2 * k) * (uu1 * c[k - 1] - uu2 * c[k - 2])
                        w = c[k] * (-z) ** (-0.5 * k)
                        t1 = (-ctx.j) ** k * ctx.mpf(2) ** (-k) * w
                        t2 = ctx.j ** k * ctx.mpf(2) ** (-k) * w
                        if abs(t1) < 0.1 * ctx.eps:
                            break
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            raise ctx.NoConvergence
                        s1 += t1
                        s2 += t2
                        tprev = t1
                        k += 1
                    S = ctx.expj(ctx.pi * X + 2 * ctx.sqrt(-z)) * s1 + ctx.expj(-(ctx.pi * X + 2 * ctx.sqrt(-z))) * s2
                    T1 = ([0.5 * S, ctx.pi, -z], [1, -0.5, X], [b1, b2], [a1], [], [], 0)
                    T2 = ([-z], [-a1], [b1, b2], [b1 - a1, b2 - a1], [a1, a1 - b1 + 1, a1 - b2 + 1], [], 1 / z)
                    return (T1, T2)
                v = ctx.hypercomb(h, [a1, b1, b2], force_series=True, maxterms=4 * ctx.prec)
                if sum((ctx._is_real_type(u) for u in [a1, b1, b2, z])) == 4:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig
    return ctx.hypsum(1, 2, (a1type, b1type, b2type), [a1, b1, b2], z, **kwargs)