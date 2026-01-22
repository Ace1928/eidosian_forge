from sympy.matrices.expressions import MatrixExpr
from sympy.assumptions.ask import Q
class RofQR(Factorization):

    @property
    def predicates(self):
        return (Q.upper_triangular,)