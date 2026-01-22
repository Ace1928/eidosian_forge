from .functions import defun, defun_wrapped
def _airy_zero(ctx, which, k, derivative, complex=False):

    def U(t):
        return t ** (2 / 3.0) * (1 - 7 / (t ** 2 * 48))

    def T(t):
        return t ** (2 / 3.0) * (1 + 5 / (t ** 2 * 48))
    k = int(k)
    if k < 1:
        raise ValueError('k cannot be less than 1')
    if not derivative in (0, 1):
        raise ValueError('Derivative should lie between 0 and 1')
    if which == 0:
        if derivative:
            return ctx.findroot(lambda z: ctx.airyai(z, 1), -U(3 * ctx.pi * (4 * k - 3) / 8))
        return ctx.findroot(ctx.airyai, -T(3 * ctx.pi * (4 * k - 1) / 8))
    if which == 1 and complex == False:
        if derivative:
            return ctx.findroot(lambda z: ctx.airybi(z, 1), -U(3 * ctx.pi * (4 * k - 1) / 8))
        return ctx.findroot(ctx.airybi, -T(3 * ctx.pi * (4 * k - 3) / 8))
    if which == 1 and complex == True:
        if derivative:
            t = 3 * ctx.pi * (4 * k - 3) / 8 + 0.75j * ctx.ln2
            s = ctx.expjpi(ctx.mpf(1) / 3) * T(t)
            return ctx.findroot(lambda z: ctx.airybi(z, 1), s)
        t = 3 * ctx.pi * (4 * k - 1) / 8 + 0.75j * ctx.ln2
        s = ctx.expjpi(ctx.mpf(1) / 3) * U(t)
        return ctx.findroot(ctx.airybi, s)