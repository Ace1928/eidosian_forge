from sympy.matrices.expressions import MatrixExpr
from sympy.assumptions.ask import Q
class EigenValues(Factorization):

    @property
    def predicates(self):
        return (Q.diagonal,)