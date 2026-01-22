from sympy.matrices.expressions import MatrixExpr
from sympy.assumptions.ask import Q
class SofSVD(Factorization):

    @property
    def predicates(self):
        return (Q.diagonal,)