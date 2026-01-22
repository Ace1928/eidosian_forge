from collections import Counter
import sys
import warnings
from time import time
from .apis import LpSolverDefault, PULP_CBC_CMD
from .apis.core import clock
from .utilities import value
from . import constants as const
from . import mps_lp as mpslp
import logging
import re
class LpElement:
    """Base class for LpVariable and LpConstraintVar"""
    illegal_chars = '-+[] ->/'
    expression = re.compile(f'[{re.escape(illegal_chars)}]')
    trans = maketrans(illegal_chars, '________')

    def setName(self, name):
        if name:
            if self.expression.match(name):
                warnings.warn('The name {} has illegal characters that will be replaced by _'.format(name))
            self.__name = str(name).translate(self.trans)
        else:
            self.__name = None

    def getName(self):
        return self.__name
    name = property(fget=getName, fset=setName)

    def __init__(self, name):
        self.name = name
        self.hash = id(self)
        self.modified = True

    def __hash__(self):
        return self.hash

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __neg__(self):
        return -LpAffineExpression(self)

    def __pos__(self):
        return self

    def __bool__(self):
        return True

    def __add__(self, other):
        return LpAffineExpression(self) + other

    def __radd__(self, other):
        return LpAffineExpression(self) + other

    def __sub__(self, other):
        return LpAffineExpression(self) - other

    def __rsub__(self, other):
        return other - LpAffineExpression(self)

    def __mul__(self, other):
        return LpAffineExpression(self) * other

    def __rmul__(self, other):
        return LpAffineExpression(self) * other

    def __div__(self, other):
        return LpAffineExpression(self) / other

    def __rdiv__(self, other):
        raise TypeError('Expressions cannot be divided by a variable')

    def __le__(self, other):
        return LpAffineExpression(self) <= other

    def __ge__(self, other):
        return LpAffineExpression(self) >= other

    def __eq__(self, other):
        return LpAffineExpression(self) == other

    def __ne__(self, other):
        if isinstance(other, LpVariable):
            return self.name is not other.name
        elif isinstance(other, LpAffineExpression):
            if other.isAtomic():
                return self is not other.atom()
            else:
                return 1
        else:
            return 1