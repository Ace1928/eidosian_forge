from .functions import defun, defun_wrapped
def _airybi_n2_inf(ctx):
    prec = ctx.prec
    try:
        v = ctx.power(3, '2/3') * ctx.gamma('2/3') / (2 * ctx.pi)
    finally:
        ctx.prec = prec
    return +v