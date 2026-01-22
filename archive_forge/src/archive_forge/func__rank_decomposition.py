import copy
from sympy.core import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.miscellaneous import Min, sqrt
from sympy.functions.elementary.complexes import sign
from .common import NonSquareMatrixError, NonPositiveDefiniteMatrixError
from .utilities import _get_intermediate_simp, _iszero
from .determinant import _find_reasonable_pivot_naive
def _rank_decomposition(M, iszerofunc=_iszero, simplify=False):
    """Returns a pair of matrices (`C`, `F`) with matching rank
    such that `A = C F`.

    Parameters
    ==========

    iszerofunc : Function, optional
        A function used for detecting whether an element can
        act as a pivot.  ``lambda x: x.is_zero`` is used by default.

    simplify : Bool or Function, optional
        A function used to simplify elements when looking for a
        pivot. By default SymPy's ``simplify`` is used.

    Returns
    =======

    (C, F) : Matrices
        `C` and `F` are full-rank matrices with rank as same as `A`,
        whose product gives `A`.

        See Notes for additional mathematical details.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [1, 3, 1, 4],
    ...     [2, 7, 3, 9],
    ...     [1, 5, 3, 1],
    ...     [1, 2, 0, 8]
    ... ])
    >>> C, F = A.rank_decomposition()
    >>> C
    Matrix([
    [1, 3, 4],
    [2, 7, 9],
    [1, 5, 1],
    [1, 2, 8]])
    >>> F
    Matrix([
    [1, 0, -2, 0],
    [0, 1,  1, 0],
    [0, 0,  0, 1]])
    >>> C * F == A
    True

    Notes
    =====

    Obtaining `F`, an RREF of `A`, is equivalent to creating a
    product

    .. math::
        E_n E_{n-1} ... E_1 A = F

    where `E_n, E_{n-1}, \\dots, E_1` are the elimination matrices or
    permutation matrices equivalent to each row-reduction step.

    The inverse of the same product of elimination matrices gives
    `C`:

    .. math::
        C = \\left(E_n E_{n-1} \\dots E_1\\right)^{-1}

    It is not necessary, however, to actually compute the inverse:
    the columns of `C` are those from the original matrix with the
    same column indices as the indices of the pivot columns of `F`.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rank_factorization

    .. [2] Piziak, R.; Odell, P. L. (1 June 1999).
        "Full Rank Factorization of Matrices".
        Mathematics Magazine. 72 (3): 193. doi:10.2307/2690882

    See Also
    ========

    sympy.matrices.matrices.MatrixReductions.rref
    """
    F, pivot_cols = M.rref(simplify=simplify, iszerofunc=iszerofunc, pivots=True)
    rank = len(pivot_cols)
    C = M.extract(range(M.rows), pivot_cols)
    F = F[:rank, :]
    return (C, F)