from types import FunctionType
from sympy.core.numbers import Float, Integer
from sympy.core.singleton import S
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.mul import Mul
from sympy.polys import PurePoly, cancel
from sympy.functions.combinatorial.numbers import nC
from sympy.polys.matrices.domainmatrix import DomainMatrix
from .common import NonSquareMatrixError
from .utilities import (
def _per(M):
    """Returns the permanent of a matrix. Unlike determinant,
    permanent is defined for both square and non-square matrices.

    For an m x n matrix, with m less than or equal to n,
    it is given as the sum over the permutations s of size
    less than or equal to m on [1, 2, . . . n] of the product
    from i = 1 to m of M[i, s[i]]. Taking the transpose will
    not affect the value of the permanent.

    In the case of a square matrix, this is the same as the permutation
    definition of the determinant, but it does not take the sign of the
    permutation into account. Computing the permanent with this definition
    is quite inefficient, so here the Ryser formula is used.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M.per()
    450
    >>> M = Matrix([1, 5, 7])
    >>> M.per()
    13

    References
    ==========

    .. [1] Prof. Frank Ben's notes: https://math.berkeley.edu/~bernd/ban275.pdf
    .. [2] Wikipedia article on Permanent: https://en.wikipedia.org/wiki/Permanent_%28mathematics%29
    .. [3] https://reference.wolfram.com/language/ref/Permanent.html
    .. [4] Permanent of a rectangular matrix : https://arxiv.org/pdf/0904.3251.pdf
    """
    import itertools
    m, n = M.shape
    if m > n:
        M = M.T
        m, n = (n, m)
    s = list(range(n))
    subsets = []
    for i in range(1, m + 1):
        subsets += list(map(list, itertools.combinations(s, i)))
    perm = 0
    for subset in subsets:
        prod = 1
        sub_len = len(subset)
        for i in range(m):
            prod *= sum([M[i, j] for j in subset])
        perm += prod * S.NegativeOne ** sub_len * nC(n - sub_len, m - sub_len)
    perm *= S.NegativeOne ** m
    return perm.simplify()