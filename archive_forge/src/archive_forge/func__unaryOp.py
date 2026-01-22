from numbers import Number
import math
import operator
import warnings
def _unaryOp(self, op):
    return self.__class__((op(v) for v in self))