from .functions import defun, defun_wrapped
@c_memo
def _airyai_C1(ctx):
    return 1 / (ctx.cbrt(9) * ctx.gamma(ctx.mpf(2) / 3))