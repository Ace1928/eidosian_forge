from .functions import defun, defun_wrapped
@c_memo
def _airybi_C1(ctx):
    return 1 / (ctx.nthroot(3, 6) * ctx.gamma(ctx.mpf(2) / 3))