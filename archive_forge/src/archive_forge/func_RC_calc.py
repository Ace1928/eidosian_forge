from .functions import defun, defun_wrapped
def RC_calc(ctx, x, y, r, pv=True):
    if not (ctx.isnormal(x) and ctx.isnormal(y)):
        if ctx.isinf(x) or ctx.isinf(y):
            return 1 / (x * y)
        if y == 0:
            return ctx.inf
        if x == 0:
            return ctx.pi / ctx.sqrt(y) / 2
        raise ValueError
    if pv and ctx._im(y) == 0 and (ctx._re(y) < 0):
        return ctx.sqrt(x / (x - y)) * RC_calc(ctx, x - y, -y, r)
    if x == y:
        return 1 / ctx.sqrt(x)
    extraprec = 2 * max(0, -ctx.mag(x - y) + ctx.mag(x))
    ctx.prec += extraprec
    if ctx._is_real_type(x) and ctx._is_real_type(y):
        x = ctx._re(x)
        y = ctx._re(y)
        a = ctx.sqrt(x / y)
        if x < y:
            b = ctx.sqrt(y - x)
            v = ctx.acos(a) / b
        else:
            b = ctx.sqrt(x - y)
            v = ctx.acosh(a) / b
    else:
        sx = ctx.sqrt(x)
        sy = ctx.sqrt(y)
        v = ctx.acos(sx / sy) / (ctx.sqrt(1 - x / y) * sy)
    ctx.prec -= extraprec
    return v