from sympy.matrices.expressions import MatrixExpr
from sympy.assumptions.ask import Q
class LofLU(Factorization):

    @property
    def predicates(self):
        return (Q.lower_triangular,)