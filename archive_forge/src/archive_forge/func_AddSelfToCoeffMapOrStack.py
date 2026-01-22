import collections
import numbers
def AddSelfToCoeffMapOrStack(self, coeffs, multiplier, stack):
    for arg in reversed(self.__array):
        stack.append((multiplier, arg))