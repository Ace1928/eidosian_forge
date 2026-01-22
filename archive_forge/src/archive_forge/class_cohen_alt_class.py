from ..libmp.backend import xrange
from .calculus import defun
class cohen_alt_class:
    """
    This interface implements the convergence acceleration of alternating series
    as described in H. Cohen, F.R. Villegas, D. Zagier - "Convergence Acceleration
    of Alternating Series". This series transformation works only well if the
    individual terms of the series have an alternating sign. It belongs to the
    class of linear series transformations (in contrast to the Shanks/Wynn-epsilon
    or Levin transform). This series transformation is also able to sum some types
    of divergent series. See the paper under which conditions this resummation is
    mathematical sound.

    Let *A* be the series we want to sum:

    .. math ::

        A = \\sum_{k=0}^{\\infty} a_k

    Let `s_n` be the partial sums of this series:

    .. math ::

        s_n = \\sum_{k=0}^n a_k.


    **Interface**

    Calling ``cohen_alt`` returns an object with the following methods.

    Then ``update(...)`` works with the list of individual terms `a_k` and
    ``update_psum(...)`` works with the list of partial sums `s_k`:

    .. code ::

        v, e = ...update([a_0, a_1,..., a_k])
        v, e = ...update_psum([s_0, s_1,..., s_k])

    *v* is the current estimate for *A*, and *e* is an error estimate which is
    simply the difference between the current estimate and the last estimate.

    **Examples**

    Here we compute the alternating zeta function using ``update_psum``::

        >>> from mpmath import mp
        >>> AC = mp.cohen_alt()
        >>> S, s, n = [], 0, 1
        >>> while 1:
        ...     s += -((-1) ** n) * mp.one / (n * n)
        ...     n += 1
        ...     S.append(s)
        ...     v, e = AC.update_psum(S)
        ...     if e < mp.eps:
        ...         break
        ...     if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> print(mp.chop(v - mp.pi ** 2 / 12))
        0.0

    Here we compute the product `\\prod_{n=1}^{\\infty} \\Gamma(1+1/(2n-1)) / \\Gamma(1+1/(2n))`::

        >>> A = []
        >>> AC = mp.cohen_alt()
        >>> n = 1
        >>> while 1:
        ...     A.append( mp.loggamma(1 + mp.one / (2 * n - 1)))
        ...     A.append(-mp.loggamma(1 + mp.one / (2 * n)))
        ...     n += 1
        ...     v, e = AC.update(A)
        ...     if e < mp.eps:
        ...         break
        ...     if n > 1000: raise RuntimeError("iteration limit exceeded")
        >>> v = mp.exp(v)
        >>> print(mp.chop(v - 1.06215090557106, tol = 1e-12))
        0.0

    ``cohen_alt`` is also accessible through the :func:`~mpmath.nsum` interface::

        >>> v = mp.nsum(lambda n: (-1)**(n-1) / n, [1, mp.inf], method = "a")
        >>> print(mp.chop(v - mp.log(2)))
        0.0
        >>> v = mp.nsum(lambda n: (-1)**n / (2 * n + 1), [0, mp.inf], method = "a")
        >>> print(mp.chop(v - mp.pi / 4))
        0.0
        >>> v = mp.nsum(lambda n: (-1)**n * mp.log(n) * n, [1, mp.inf], method = "a")
        >>> print(mp.chop(v - mp.diff(lambda s: mp.altzeta(s), -1)))
        0.0

    """

    def __init__(self):
        self.last = 0

    def update(self, A):
        """
        This routine applies the convergence acceleration to the list of individual terms.

        A    = sum(a_k, k = 0..infinity)

        v, e = ...update([a_0, a_1,..., a_k])

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """
        n = len(A)
        d = (3 + self.ctx.sqrt(8)) ** n
        d = (d + 1 / d) / 2
        b = -self.ctx.one
        c = -d
        s = 0
        for k in xrange(n):
            c = b - c
            if k % 2 == 0:
                s = s + c * A[k]
            else:
                s = s - c * A[k]
            b = 2 * (k + n) * (k - n) * b / ((2 * k + 1) * (k + self.ctx.one))
        value = s / d
        err = abs(value - self.last)
        self.last = value
        return (value, err)

    def update_psum(self, S):
        """
        This routine applies the convergence acceleration to the list of partial sums.

        A   = sum(a_k, k = 0..infinity)
        s_n = sum(a_k ,k = 0..n)

        v, e = ...update_psum([s_0, s_1,..., s_k])

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """
        n = len(S)
        d = (3 + self.ctx.sqrt(8)) ** n
        d = (d + 1 / d) / 2
        b = self.ctx.one
        s = 0
        for k in xrange(n):
            b = 2 * (n + k) * (n - k) * b / ((2 * k + 1) * (k + self.ctx.one))
            s += b * S[k]
        value = s / d
        err = abs(value - self.last)
        self.last = value
        return (value, err)