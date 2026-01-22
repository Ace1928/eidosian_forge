from sympy.matrices.expressions import MatrixExpr
from sympy.assumptions.ask import Q
class Factorization(MatrixExpr):
    arg = property(lambda self: self.args[0])
    shape = property(lambda self: self.arg.shape)