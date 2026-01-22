from .functions import defun, defun_wrapped
@defun
def _jacobi_theta3a(ctx, z, q):
    """
    case ctx._im(z) != 0
    theta3(z, q) = Sum(q**(n*n) * exp(j*2*n*z), n, -inf, inf)
    max term for n*abs(log(q).real) + ctx._im(z) ~= 0
    n0 = int(- ctx._im(z)/abs(log(q).real))
    """
    n = n0 = int(-ctx._im(z) / abs(ctx._re(ctx.log(q))))
    e2 = ctx.expj(2 * z)
    e = e0 = ctx.expj(2 * n * z)
    s = term = q ** (n * n) * e
    eps1 = ctx.eps * abs(term)
    while 1:
        n += 1
        e = e * e2
        term = q ** (n * n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2 * z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = q ** (n * n) * e
        if abs(term) < eps1:
            break
        s += term
    return s