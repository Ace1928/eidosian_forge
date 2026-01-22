from .functions import defun, defun_wrapped
@defun
def _djtheta(ctx, n, z, q, derivative=1):
    z = ctx.convert(z)
    q = ctx.convert(q)
    nd = int(derivative)
    if abs(q) > ctx.THETA_Q_LIM:
        raise ValueError('abs(q) > THETA_Q_LIM = %f' % ctx.THETA_Q_LIM)
    extra = 10 + ctx.prec * nd // 10
    if z:
        M = ctx.mag(z)
        if M > 5 or (n != 1 and M < -5):
            extra += 2 * abs(M)
    cz = 0.5
    extra2 = 50
    prec0 = ctx.prec
    try:
        ctx.prec += extra
        if n == 1:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta2(z - ctx.pi / 2, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta2a(z - ctx.pi / 2, q, nd)
            else:
                res = ctx._djacobi_theta2(z - ctx.pi / 2, q, nd)
        elif n == 2:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta2(z, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta2a(z, q, nd)
            else:
                res = ctx._djacobi_theta2(z, q, nd)
        elif n == 3:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta3(z, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta3a(z, q, nd)
            else:
                res = ctx._djacobi_theta3(z, q, nd)
        elif n == 4:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta3(z, -q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta3a(z, -q, nd)
            else:
                res = ctx._djacobi_theta3(z, -q, nd)
        else:
            raise ValueError
    finally:
        ctx.prec = prec0
    return +res