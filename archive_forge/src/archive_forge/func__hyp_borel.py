from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def _hyp_borel(ctx, p, q, a_s, b_s, z, **kwargs):
    if a_s:
        a_s, a_types = zip(*a_s)
        a_s = list(a_s)
    else:
        a_s, a_types = ([], ())
    if b_s:
        b_s, b_types = zip(*b_s)
        b_s = list(b_s)
    else:
        b_s, b_types = ([], ())
    kwargs['maxterms'] = kwargs.get('maxterms', ctx.prec)
    try:
        return ctx.hypsum(p, q, a_types + b_types, a_s + b_s, z, **kwargs)
    except ctx.NoConvergence:
        pass
    prec = ctx.prec
    try:
        tol = kwargs.get('asymp_tol', ctx.eps / 4)
        ctx.prec += 10

        def term(k, cache={0: ctx.one}):
            if k in cache:
                return cache[k]
            t = term(k - 1)
            for a in a_s:
                t *= a + (k - 1)
            for b in b_s:
                t /= b + (k - 1)
            t *= z
            t /= k
            cache[k] = t
            return t
        s = ctx.one
        for k in xrange(1, ctx.prec):
            t = term(k)
            s += t
            if abs(t) <= tol:
                return s
    finally:
        ctx.prec = prec
    if p <= q + 3:
        contour = kwargs.get('contour')
        if not contour:
            if ctx.arg(z) < 0.25:
                u = z / max(1, abs(z))
                if ctx.arg(z) >= 0:
                    contour = [0, 2j, (2j + 2) / u, 2 / u, ctx.inf]
                else:
                    contour = [0, -2j, (-2j + 2) / u, 2 / u, ctx.inf]
            else:
                contour = [0, ctx.inf]
        quad_kwargs = kwargs.get('quad_kwargs', {})

        def g(t):
            return ctx.exp(-t) * ctx.hyper(a_s, b_s + [1], t * z)
        I, err = ctx.quad(g, contour, error=True, **quad_kwargs)
        if err <= abs(I) * ctx.eps * 8:
            return I
    raise ctx.NoConvergence