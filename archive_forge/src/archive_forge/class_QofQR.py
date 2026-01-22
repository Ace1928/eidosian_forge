from sympy.matrices.expressions import MatrixExpr
from sympy.assumptions.ask import Q
class QofQR(Factorization):

    @property
    def predicates(self):
        return (Q.orthogonal,)