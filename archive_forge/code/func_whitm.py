from .functions import defun, defun_wrapped
@defun_wrapped
def whitm(ctx, k, m, z, **kwargs):
    if z == 0:
        if ctx.re(m) > -0.5:
            return z
        elif ctx.re(m) < -0.5:
            return ctx.inf + z
        else:
            return ctx.nan * z
    x = ctx.fmul(-0.5, z, exact=True)
    y = 0.5 + m
    return ctx.exp(x) * z ** y * ctx.hyp1f1(y - k, 1 + 2 * m, z, **kwargs)