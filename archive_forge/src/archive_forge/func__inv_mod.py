from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _inv_mod(M, m):
    """
    Returns the inverse of the matrix `K` (mod `m`), if it exists.

    Method to find the matrix inverse of `K` (mod `m`) implemented in this function:

    * Compute `\\mathrm{adj}(K) = \\mathrm{cof}(K)^t`, the adjoint matrix of `K`.

    * Compute `r = 1/\\mathrm{det}(K) \\pmod m`.

    * `K^{-1} = r\\cdot \\mathrm{adj}(K) \\pmod m`.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix(2, 2, [1, 2, 3, 4])
    >>> A.inv_mod(5)
    Matrix([
    [3, 1],
    [4, 2]])
    >>> A.inv_mod(3)
    Matrix([
    [1, 1],
    [0, 1]])

    """
    if not M.is_square:
        raise NonSquareMatrixError()
    N = M.cols
    det_K = M.det()
    det_inv = None
    try:
        det_inv = mod_inverse(det_K, m)
    except ValueError:
        raise NonInvertibleMatrixError('Matrix is not invertible (mod %d)' % m)
    K_adj = M.adjugate()
    K_inv = M.__class__(N, N, [det_inv * K_adj[i, j] % m for i in range(N) for j in range(N)])
    return K_inv