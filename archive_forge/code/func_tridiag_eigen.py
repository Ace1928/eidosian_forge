from ..libmp.backend import xrange
from .eigen import defun
def tridiag_eigen(ctx, d, e, z=False):
    """
    This subroutine find the eigenvalues and the first components of the
    eigenvectors of a real symmetric tridiagonal matrix using the implicit
    QL method.

    parameters:

      d (input/output) real array of length n. on input, d contains the diagonal
        elements of the input matrix. on output, d contains the eigenvalues in
        ascending order.

      e (input) real array of length n. on input, e contains the offdiagonal
        elements of the input matrix in e[0:(n-1)]. On output, e has been
        destroyed.

      z (input/output) If z is equal to False, no eigenvectors will be computed.
        Otherwise on input z should have the format z[0:m,0:n] (i.e. a real or
        complex matrix of dimension (m,n) ). On output this matrix will be
        multiplied by the matrix of the eigenvectors (i.e. the columns of this
        matrix are the eigenvectors): z --> z*EV
        That means if z[i,j]={1 if j==j; 0 otherwise} on input, then on output
        z will contain the first m components of the eigenvectors. That means
        if m is equal to n, the i-th eigenvector will be z[:,i].

    This routine is a python translation (in slightly modified form) of the
    fortran routine imtql2.f in the software library EISPACK (see netlib.org)
    which itself is based on the algol procudure imtql2 desribed in:
     - num. math. 12, p. 377-383(1968) by matrin and wilkinson
     - modified in num. math. 15, p. 450(1970) by dubrulle
     - handbook for auto. comp., vol. II-linear algebra, p. 241-248 (1971)
    See also the routine gaussq.f in netlog.org or acm algorithm 726.
    """
    n = len(d)
    e[n - 1] = 0
    iterlim = 2 * ctx.dps
    for l in xrange(n):
        j = 0
        while 1:
            m = l
            while 1:
                if m + 1 == n:
                    break
                if abs(e[m]) <= ctx.eps * (abs(d[m]) + abs(d[m + 1])):
                    break
                m = m + 1
            if m == l:
                break
            if j >= iterlim:
                raise RuntimeError('tridiag_eigen: no convergence to an eigenvalue after %d iterations' % iterlim)
            j += 1
            p = d[l]
            g = (d[l + 1] - p) / (2 * e[l])
            r = ctx.hypot(g, 1)
            if g < 0:
                s = g - r
            else:
                s = g + r
            g = d[m] - p + e[l] / s
            s, c, p = (1, 1, 0)
            for i in xrange(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if abs(f) > abs(g):
                    c = g / f
                    r = ctx.hypot(c, 1)
                    e[i + 1] = f * r
                    s = 1 / r
                    c = c * s
                else:
                    s = f / g
                    r = ctx.hypot(s, 1)
                    e[i + 1] = g * r
                    c = 1 / r
                    s = s * c
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                if not isinstance(z, bool):
                    for w in xrange(z.rows):
                        f = z[w, i + 1]
                        z[w, i + 1] = s * z[w, i] + c * f
                        z[w, i] = c * z[w, i] - s * f
            d[l] = d[l] - p
            e[l] = g
            e[m] = 0
    for ii in xrange(1, n):
        i = ii - 1
        k = i
        p = d[i]
        for j in xrange(ii, n):
            if d[j] >= p:
                continue
            k = j
            p = d[k]
        if k == i:
            continue
        d[k] = d[i]
        d[i] = p
        if not isinstance(z, bool):
            for w in xrange(z.rows):
                p = z[w, i]
                z[w, i] = z[w, k]
                z[w, k] = p