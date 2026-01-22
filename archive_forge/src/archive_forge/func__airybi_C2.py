from .functions import defun, defun_wrapped
@c_memo
def _airybi_C2(ctx):
    return ctx.nthroot(3, 6) / ctx.gamma(ctx.mpf(1) / 3)