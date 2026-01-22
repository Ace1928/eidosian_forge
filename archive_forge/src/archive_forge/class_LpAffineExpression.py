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
class LpAffineExpression(_DICT_TYPE):
    """
    A linear combination of :class:`LpVariables<LpVariable>`.
    Can be initialised with the following:

    #.   e = None: an empty Expression
    #.   e = dict: gives an expression with the values being the coefficients of the keys (order of terms is undetermined)
    #.   e = list or generator of 2-tuples: equivalent to dict.items()
    #.   e = LpElement: an expression of length 1 with the coefficient 1
    #.   e = other: the constant is initialised as e

    Examples:

       >>> f=LpAffineExpression(LpElement('x'))
       >>> f
       1*x + 0
       >>> x_name = ['x_0', 'x_1', 'x_2']
       >>> x = [LpVariable(x_name[i], lowBound = 0, upBound = 10) for i in range(3) ]
       >>> c = LpAffineExpression([ (x[0],1), (x[1],-3), (x[2],4)])
       >>> c
       1*x_0 + -3*x_1 + 4*x_2 + 0
    """
    trans = maketrans('-+[] ', '_____')

    def setName(self, name):
        if name:
            self.__name = str(name).translate(self.trans)
        else:
            self.__name = None

    def getName(self):
        return self.__name
    name = property(fget=getName, fset=setName)

    def __init__(self, e=None, constant=0, name=None):
        self.name = name
        if e is None:
            e = {}
        if isinstance(e, LpAffineExpression):
            self.constant = e.constant
            super().__init__(list(e.items()))
        elif isinstance(e, dict):
            self.constant = constant
            super().__init__(list(e.items()))
        elif isinstance(e, Iterable):
            self.constant = constant
            super().__init__(e)
        elif isinstance(e, LpElement):
            self.constant = 0
            super().__init__([(e, 1)])
        else:
            self.constant = e
            super().__init__()

    def isAtomic(self):
        return len(self) == 1 and self.constant == 0 and (list(self.values())[0] == 1)

    def isNumericalConstant(self):
        return len(self) == 0

    def atom(self):
        return list(self.keys())[0]

    def __bool__(self):
        return float(self.constant) != 0.0 or len(self) > 0

    def value(self):
        s = self.constant
        for v, x in self.items():
            if v.varValue is None:
                return None
            s += v.varValue * x
        return s

    def valueOrDefault(self):
        s = self.constant
        for v, x in self.items():
            s += v.valueOrDefault() * x
        return s

    def addterm(self, key, value):
        y = self.get(key, 0)
        if y:
            y += value
            self[key] = y
        else:
            self[key] = value

    def emptyCopy(self):
        return LpAffineExpression()

    def copy(self):
        """Make a copy of self except the name which is reset"""
        return LpAffineExpression(self)

    def __str__(self, constant=1):
        s = ''
        for v in self.sorted_keys():
            val = self[v]
            if val < 0:
                if s != '':
                    s += ' - '
                else:
                    s += '-'
                val = -val
            elif s != '':
                s += ' + '
            if val == 1:
                s += str(v)
            else:
                s += str(val) + '*' + str(v)
        if constant:
            if s == '':
                s = str(self.constant)
            elif self.constant < 0:
                s += ' - ' + str(-self.constant)
            elif self.constant > 0:
                s += ' + ' + str(self.constant)
        elif s == '':
            s = '0'
        return s

    def sorted_keys(self):
        """
        returns the list of keys sorted by name
        """
        result = [(v.name, v) for v in self.keys()]
        result.sort()
        result = [v for _, v in result]
        return result

    def __repr__(self):
        l = [str(self[v]) + '*' + str(v) for v in self.sorted_keys()]
        l.append(str(self.constant))
        s = ' + '.join(l)
        return s

    @staticmethod
    def _count_characters(line):
        return sum((len(t) for t in line))

    def asCplexVariablesOnly(self, name):
        """
        helper for asCplexLpAffineExpression
        """
        result = []
        line = [f'{name}:']
        notFirst = 0
        variables = self.sorted_keys()
        for v in variables:
            val = self[v]
            if val < 0:
                sign = ' -'
                val = -val
            elif notFirst:
                sign = ' +'
            else:
                sign = ''
            notFirst = 1
            if val == 1:
                term = f'{sign} {v.name}'
            else:
                term = f'{sign} {val + 0:.12g} {v.name}'
            if self._count_characters(line) + len(term) > const.LpCplexLPLineSize:
                result += [''.join(line)]
                line = [term]
            else:
                line += [term]
        return (result, line)

    def asCplexLpAffineExpression(self, name, constant=1):
        """
        returns a string that represents the Affine Expression in lp format
        """
        result, line = self.asCplexVariablesOnly(name)
        if not self:
            term = f' {self.constant}'
        else:
            term = ''
            if constant:
                if self.constant < 0:
                    term = ' - %s' % -self.constant
                elif self.constant > 0:
                    term = f' + {self.constant}'
        if self._count_characters(line) + len(term) > const.LpCplexLPLineSize:
            result += [''.join(line)]
            line = [term]
        else:
            line += [term]
        result += [''.join(line)]
        result = '%s\n' % '\n'.join(result)
        return result

    def addInPlace(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if other is None:
            return self
        if isinstance(other, LpElement):
            self.addterm(other, 1)
        elif isinstance(other, LpAffineExpression):
            self.constant += other.constant
            for v, x in other.items():
                self.addterm(v, x)
        elif isinstance(other, dict):
            for e in other.values():
                self.addInPlace(e)
        elif isinstance(other, list) or isinstance(other, Iterable):
            for e in other:
                self.addInPlace(e)
        else:
            self.constant += other
        return self

    def subInPlace(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if other is None:
            return self
        if isinstance(other, LpElement):
            self.addterm(other, -1)
        elif isinstance(other, LpAffineExpression):
            self.constant -= other.constant
            for v, x in other.items():
                self.addterm(v, -x)
        elif isinstance(other, dict):
            for e in other.values():
                self.subInPlace(e)
        elif isinstance(other, list) or isinstance(other, Iterable):
            for e in other:
                self.subInPlace(e)
        else:
            self.constant -= other
        return self

    def __neg__(self):
        e = self.emptyCopy()
        e.constant = -self.constant
        for v, x in self.items():
            e[v] = -x
        return e

    def __pos__(self):
        return self

    def __add__(self, other):
        return self.copy().addInPlace(other)

    def __radd__(self, other):
        return self.copy().addInPlace(other)

    def __iadd__(self, other):
        return self.addInPlace(other)

    def __sub__(self, other):
        return self.copy().subInPlace(other)

    def __rsub__(self, other):
        return (-self).addInPlace(other)

    def __isub__(self, other):
        return self.subInPlace(other)

    def __mul__(self, other):
        e = self.emptyCopy()
        if isinstance(other, LpAffineExpression):
            e.constant = self.constant * other.constant
            if len(other):
                if len(self):
                    raise TypeError('Non-constant expressions cannot be multiplied')
                else:
                    c = self.constant
                    if c != 0:
                        for v, x in other.items():
                            e[v] = c * x
            else:
                c = other.constant
                if c != 0:
                    for v, x in self.items():
                        e[v] = c * x
        elif isinstance(other, LpVariable):
            return self * LpAffineExpression(other)
        elif other != 0:
            e.constant = self.constant * other
            for v, x in self.items():
                e[v] = other * x
        return e

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other, LpAffineExpression) or isinstance(other, LpVariable):
            if len(other):
                raise TypeError('Expressions cannot be divided by a non-constant expression')
            other = other.constant
        e = self.emptyCopy()
        e.constant = self.constant / other
        for v, x in self.items():
            e[v] = x / other
        return e

    def __truediv__(self, other):
        if isinstance(other, LpAffineExpression) or isinstance(other, LpVariable):
            if len(other):
                raise TypeError('Expressions cannot be divided by a non-constant expression')
            other = other.constant
        e = self.emptyCopy()
        e.constant = self.constant / other
        for v, x in self.items():
            e[v] = x / other
        return e

    def __rdiv__(self, other):
        e = self.emptyCopy()
        if len(self):
            raise TypeError('Expressions cannot be divided by a non-constant expression')
        c = self.constant
        if isinstance(other, LpAffineExpression):
            e.constant = other.constant / c
            for v, x in other.items():
                e[v] = x / c
        else:
            e.constant = other / c
        return e

    def __le__(self, other):
        return LpConstraint(self - other, const.LpConstraintLE)

    def __ge__(self, other):
        return LpConstraint(self - other, const.LpConstraintGE)

    def __eq__(self, other):
        return LpConstraint(self - other, const.LpConstraintEQ)

    def toDict(self):
        """
        exports the :py:class:`LpAffineExpression` into a list of dictionaries with the coefficients
        it does not export the constant

        :return: list of dictionaries with the coefficients
        :rtype: list
        """
        return [dict(name=k.name, value=v) for k, v in self.items()]
    to_dict = toDict