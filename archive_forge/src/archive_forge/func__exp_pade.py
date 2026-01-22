from ..libmp.backend import xrange
def _exp_pade(ctx, a):
    """
        Exponential of a matrix using Pade approximants.

        See G. H. Golub, C. F. van Loan 'Matrix Computations',
        third Ed., page 572

        TODO:
         - find a good estimate for q
         - reduce the number of matrix multiplications to improve
           performance
        """

    def eps_pade(p):
        return ctx.mpf(2) ** (3 - 2 * p) * ctx.factorial(p) ** 2 / (ctx.factorial(2 * p) ** 2 * (2 * p + 1))
    q = 4
    extraq = 8
    while 1:
        if eps_pade(q) < ctx.eps:
            break
        q += 1
    q += extraq
    j = int(max(1, ctx.mag(ctx.mnorm(a, 'inf'))))
    extra = q
    prec = ctx.prec
    ctx.dps += extra + 3
    try:
        a = a / 2 ** j
        na = a.rows
        den = ctx.eye(na)
        num = ctx.eye(na)
        x = ctx.eye(na)
        c = ctx.mpf(1)
        for k in range(1, q + 1):
            c *= ctx.mpf(q - k + 1) / ((2 * q - k + 1) * k)
            x = a * x
            cx = c * x
            num += cx
            den += (-1) ** k * cx
        f = ctx.lu_solve_mat(den, num)
        for k in range(j):
            f = f * f
    finally:
        ctx.prec = prec
    return f * 1