from .functions import defun, defun_wrapped
@defun
def _lower_gamma(ctx, z, b, regularized=False):
    if ctx.isnpint(z):
        return type(z)(ctx.inf)
    G = [z] * regularized
    negb = ctx.fneg(b, exact=True)

    def h(z):
        T1 = ([ctx.exp(negb), b, z], [1, z, -1], [], G, [1], [1 + z], b)
        return (T1,)
    return ctx.hypercomb(h, [z])