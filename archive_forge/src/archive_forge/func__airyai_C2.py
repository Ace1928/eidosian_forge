from .functions import defun, defun_wrapped
@c_memo
def _airyai_C2(ctx):
    return -1 / (ctx.cbrt(3) * ctx.gamma(ctx.mpf(1) / 3))