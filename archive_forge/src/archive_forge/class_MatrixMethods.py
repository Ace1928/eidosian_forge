from ..libmp.backend import xrange
import warnings
class MatrixMethods(object):

    def __init__(ctx):
        ctx.matrix = type('matrix', (_matrix,), {})
        ctx.matrix.ctx = ctx
        ctx.matrix.convert = ctx.convert

    def eye(ctx, n, **kwargs):
        """
        Create square identity matrix n x n.
        """
        A = ctx.matrix(n, **kwargs)
        for i in xrange(n):
            A[i, i] = 1
        return A

    def diag(ctx, diagonal, **kwargs):
        """
        Create square diagonal matrix using given list.

        Example:
        >>> from mpmath import diag, mp
        >>> mp.pretty = False
        >>> diag([1, 2, 3])
        matrix(
        [['1.0', '0.0', '0.0'],
         ['0.0', '2.0', '0.0'],
         ['0.0', '0.0', '3.0']])
        """
        A = ctx.matrix(len(diagonal), **kwargs)
        for i in xrange(len(diagonal)):
            A[i, i] = diagonal[i]
        return A

    def zeros(ctx, *args, **kwargs):
        """
        Create matrix m x n filled with zeros.
        One given dimension will create square matrix n x n.

        Example:
        >>> from mpmath import zeros, mp
        >>> mp.pretty = False
        >>> zeros(2)
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])
        """
        if len(args) == 1:
            m = n = args[0]
        elif len(args) == 2:
            m = args[0]
            n = args[1]
        else:
            raise TypeError('zeros expected at most 2 arguments, got %i' % len(args))
        A = ctx.matrix(m, n, **kwargs)
        for i in xrange(m):
            for j in xrange(n):
                A[i, j] = 0
        return A

    def ones(ctx, *args, **kwargs):
        """
        Create matrix m x n filled with ones.
        One given dimension will create square matrix n x n.

        Example:
        >>> from mpmath import ones, mp
        >>> mp.pretty = False
        >>> ones(2)
        matrix(
        [['1.0', '1.0'],
         ['1.0', '1.0']])
        """
        if len(args) == 1:
            m = n = args[0]
        elif len(args) == 2:
            m = args[0]
            n = args[1]
        else:
            raise TypeError('ones expected at most 2 arguments, got %i' % len(args))
        A = ctx.matrix(m, n, **kwargs)
        for i in xrange(m):
            for j in xrange(n):
                A[i, j] = 1
        return A

    def hilbert(ctx, m, n=None):
        """
        Create (pseudo) hilbert matrix m x n.
        One given dimension will create hilbert matrix n x n.

        The matrix is very ill-conditioned and symmetric, positive definite if
        square.
        """
        if n is None:
            n = m
        A = ctx.matrix(m, n)
        for i in xrange(m):
            for j in xrange(n):
                A[i, j] = ctx.one / (i + j + 1)
        return A

    def randmatrix(ctx, m, n=None, min=0, max=1, **kwargs):
        """
        Create a random m x n matrix.

        All values are >= min and <max.
        n defaults to m.

        Example:
        >>> from mpmath import randmatrix
        >>> randmatrix(2) # doctest:+SKIP
        matrix(
        [['0.53491598236191806', '0.57195669543302752'],
         ['0.85589992269513615', '0.82444367501382143']])
        """
        if not n:
            n = m
        A = ctx.matrix(m, n, **kwargs)
        for i in xrange(m):
            for j in xrange(n):
                A[i, j] = ctx.rand() * (max - min) + min
        return A

    def swap_row(ctx, A, i, j):
        """
        Swap row i with row j.
        """
        if i == j:
            return
        if isinstance(A, ctx.matrix):
            for k in xrange(A.cols):
                A[i, k], A[j, k] = (A[j, k], A[i, k])
        elif isinstance(A, list):
            A[i], A[j] = (A[j], A[i])
        else:
            raise TypeError('could not interpret type')

    def extend(ctx, A, b):
        """
        Extend matrix A with column b and return result.
        """
        if not isinstance(A, ctx.matrix):
            raise TypeError('A should be a type of ctx.matrix')
        if A.rows != len(b):
            raise ValueError('Value should be equal to len(b)')
        A = A.copy()
        A.cols += 1
        for i in xrange(A.rows):
            A[i, A.cols - 1] = b[i]
        return A

    def norm(ctx, x, p=2):
        """
        Gives the entrywise `p`-norm of an iterable *x*, i.e. the vector norm
        `\\left(\\sum_k |x_k|^p\\right)^{1/p}`, for any given `1 \\le p \\le \\infty`.

        Special cases:

        If *x* is not iterable, this just returns ``absmax(x)``.

        ``p=1`` gives the sum of absolute values.

        ``p=2`` is the standard Euclidean vector norm.

        ``p=inf`` gives the magnitude of the largest element.

        For *x* a matrix, ``p=2`` is the Frobenius norm.
        For operator matrix norms, use :func:`~mpmath.mnorm` instead.

        You can use the string 'inf' as well as float('inf') or mpf('inf')
        to specify the infinity norm.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> x = matrix([-10, 2, 100])
            >>> norm(x, 1)
            mpf('112.0')
            >>> norm(x, 2)
            mpf('100.5186549850325')
            >>> norm(x, inf)
            mpf('100.0')

        """
        try:
            iter(x)
        except TypeError:
            return ctx.absmax(x)
        if type(p) is not int:
            p = ctx.convert(p)
        if p == ctx.inf:
            return max((ctx.absmax(i) for i in x))
        elif p == 1:
            return ctx.fsum(x, absolute=1)
        elif p == 2:
            return ctx.sqrt(ctx.fsum(x, absolute=1, squared=1))
        elif p > 1:
            return ctx.nthroot(ctx.fsum((abs(i) ** p for i in x)), p)
        else:
            raise ValueError('p has to be >= 1')

    def mnorm(ctx, A, p=1):
        """
        Gives the matrix (operator) `p`-norm of A. Currently ``p=1`` and ``p=inf``
        are supported:

        ``p=1`` gives the 1-norm (maximal column sum)

        ``p=inf`` gives the `\\infty`-norm (maximal row sum).
        You can use the string 'inf' as well as float('inf') or mpf('inf')

        ``p=2`` (not implemented) for a square matrix is the usual spectral
        matrix norm, i.e. the largest singular value.

        ``p='f'`` (or 'F', 'fro', 'Frobenius, 'frobenius') gives the
        Frobenius norm, which is the elementwise 2-norm. The Frobenius norm is an
        approximation of the spectral norm and satisfies

        .. math ::

            \\frac{1}{\\sqrt{\\mathrm{rank}(A)}} \\|A\\|_F \\le \\|A\\|_2 \\le \\|A\\|_F

        The Frobenius norm lacks some mathematical properties that might
        be expected of a norm.

        For general elementwise `p`-norms, use :func:`~mpmath.norm` instead.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> A = matrix([[1, -1000], [100, 50]])
            >>> mnorm(A, 1)
            mpf('1050.0')
            >>> mnorm(A, inf)
            mpf('1001.0')
            >>> mnorm(A, 'F')
            mpf('1006.2310867787777')

        """
        A = ctx.matrix(A)
        if type(p) is not int:
            if type(p) is str and 'frobenius'.startswith(p.lower()):
                return ctx.norm(A, 2)
            p = ctx.convert(p)
        m, n = (A.rows, A.cols)
        if p == 1:
            return max((ctx.fsum((A[i, j] for i in xrange(m)), absolute=1) for j in xrange(n)))
        elif p == ctx.inf:
            return max((ctx.fsum((A[i, j] for j in xrange(n)), absolute=1) for i in xrange(m)))
        else:
            raise NotImplementedError('matrix p-norm for arbitrary p')