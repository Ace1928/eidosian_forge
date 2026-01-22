from ..libmp.backend import xrange
import math
import cmath
def _lambertw_series(ctx, z, k, tol):
    """
    Return rough approximation for W_k(z) from an asymptotic series,
    sufficiently accurate for the Halley iteration to converge to
    the correct value.
    """
    magz = ctx.mag(z)
    if -10 < magz < 900 and -1000 < k < 1000:
        if magz < 1 and abs(z + 0.36787944117144) < 0.05:
            if k == 0 or (k == -1 and ctx._im(z) >= 0) or (k == 1 and ctx._im(z) < 0):
                delta = ctx.sum_accurately(lambda: [z, ctx.exp(-1)])
                cancellation = -ctx.mag(delta)
                ctx.prec += cancellation
                p = ctx.sqrt(2 * (ctx.e * z + 1))
                ctx.prec -= cancellation
                u = {0: ctx.mpf(-1), 1: ctx.mpf(1)}
                a = {0: ctx.mpf(2), 1: ctx.mpf(-1)}
                if k != 0:
                    p = -p
                s = ctx.zero
                for l in xrange(max(2, cancellation)):
                    if l not in u:
                        a[l] = ctx.fsum((u[j] * u[l + 1 - j] for j in xrange(2, l)))
                        u[l] = (l - 1) * (u[l - 2] / 2 + a[l - 2] / 4) / (l + 1) - a[l] / 2 - u[l - 1] / (l + 1)
                    term = u[l] * p ** l
                    s += term
                    if ctx.mag(term) < -tol:
                        return (s, True)
                    l += 1
                ctx.prec += cancellation // 2
                return (s, False)
        if k == 0 or k == -1:
            return (_lambertw_approx_hybrid(z, k), False)
    if k == 0:
        if magz < -1:
            return (z * (1 - z), False)
        L1 = ctx.ln(z)
        L2 = ctx.ln(L1)
    elif k == -1 and (not ctx._im(z)) and (-0.36787944117144 < ctx._re(z) < 0):
        L1 = ctx.ln(-z)
        return (L1 - ctx.ln(-L1), False)
    else:
        L1 = ctx.ln(z) + 2j * ctx.pi * k
        L2 = ctx.ln(L1)
    return (L1 - L2 + L2 / L1 + L2 * (L2 - 2) / (2 * L1 ** 2), False)