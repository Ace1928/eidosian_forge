from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.matrices.common import NonSquareMatrixError
from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
class Determinant(Expr):
    """Matrix Determinant

    Represents the determinant of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Determinant, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Determinant(A)
    Determinant(A)
    >>> Determinant(eye(3)).doit()
    1
    """
    is_commutative = True

    def __new__(cls, mat):
        mat = sympify(mat)
        if not mat.is_Matrix:
            raise TypeError('Input to Determinant, %s, not a matrix' % str(mat))
        if mat.is_square is False:
            raise NonSquareMatrixError('Det of a non-square matrix')
        return Basic.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]

    @property
    def kind(self):
        return self.arg.kind.element_kind

    def doit(self, expand=False, **hints):
        try:
            return self.arg._eval_determinant()
        except (AttributeError, NotImplementedError):
            return self