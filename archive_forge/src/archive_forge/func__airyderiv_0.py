from .functions import defun, defun_wrapped
def _airyderiv_0(ctx, z, n, ntype, which):
    if ntype == 'Z':
        if n < 0:
            return z
        r = ctx.mpq_1_3
        prec = ctx.prec
        try:
            ctx.prec += 10
            v = ctx.gamma((n + 1) * r) * ctx.power(3, n * r) / ctx.pi
            if which == 0:
                v *= ctx.sinpi(2 * (n + 1) * r)
                v /= ctx.power(3, '2/3')
            else:
                v *= abs(ctx.sinpi(2 * (n + 1) * r))
                v /= ctx.power(3, '1/6')
        finally:
            ctx.prec = prec
        return +v + z
    else:
        raise NotImplementedError