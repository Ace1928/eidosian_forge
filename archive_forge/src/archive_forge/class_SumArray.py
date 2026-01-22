import collections
import numbers
class SumArray(LinearExpr):
    """Represents the sum of a list of LinearExpr."""

    def __init__(self, array):
        self.__array = [CastToLinExp(elem) for elem in array]

    def __str__(self):
        return '({})'.format(' + '.join(map(str, self.__array)))

    def AddSelfToCoeffMapOrStack(self, coeffs, multiplier, stack):
        for arg in reversed(self.__array):
            stack.append((multiplier, arg))