from sympy.core.sympify import _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
class IDFT(DFT):
    """
    Returns an inverse discrete Fourier transform matrix. The matrix is scaled
    with :math:`\\frac{1}{\\sqrt{n}}` so that it is unitary.

    Parameters
    ==========

    n : integer or Symbol
        Size of the transform

    Examples
    ========

    >>> from sympy.matrices.expressions.fourier import DFT, IDFT
    >>> IDFT(3)
    IDFT(3)
    >>> IDFT(4)*DFT(4)
    I

    See Also
    ========

    DFT

    """

    def _entry(self, i, j, **kwargs):
        w = exp(-2 * S.Pi * I / self.n)
        return w ** (-i * j) / sqrt(self.n)

    def _eval_inverse(self):
        return DFT(self.n)