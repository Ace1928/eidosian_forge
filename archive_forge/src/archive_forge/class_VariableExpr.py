import collections
import numbers
class VariableExpr(LinearExpr):
    """Represents a LinearExpr containing only a single variable."""

    def __init__(self, mpvar):
        self.__var = mpvar

    def AddSelfToCoeffMapOrStack(self, coeffs, multiplier, stack):
        coeffs[self.__var] += multiplier