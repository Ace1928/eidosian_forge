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
class LpVariable(LpElement):
    """
    This class models an LP Variable with the specified associated parameters

    :param name: The name of the variable used in the output .lp file
    :param lowBound: The lower bound on this variable's range.
        Default is negative infinity
    :param upBound: The upper bound on this variable's range.
        Default is positive infinity
    :param cat: The category this variable is in, Integer, Binary or
        Continuous(default)
    :param e: Used for column based modelling: relates to the variable's
        existence in the objective function and constraints
    """

    def __init__(self, name, lowBound=None, upBound=None, cat=const.LpContinuous, e=None):
        LpElement.__init__(self, name)
        self._lowbound_original = self.lowBound = lowBound
        self._upbound_original = self.upBound = upBound
        self.cat = cat
        self.varValue = None
        self.dj = None
        if cat == const.LpBinary:
            self._lowbound_original = self.lowBound = 0
            self._upbound_original = self.upBound = 1
            self.cat = const.LpInteger
        if e:
            self.add_expression(e)

    def toDict(self):
        """
        Exports a variable into a dictionary with its relevant information

        :return: a dictionary with the variable information
        :rtype: dict
        """
        return dict(lowBound=self.lowBound, upBound=self.upBound, cat=self.cat, varValue=self.varValue, dj=self.dj, name=self.name)
    to_dict = toDict

    @classmethod
    def fromDict(cls, dj=None, varValue=None, **kwargs):
        """
        Initializes a variable object from information that comes from a dictionary (kwargs)

        :param dj: shadow price of the variable
        :param float varValue: the value to set the variable
        :param kwargs: arguments to initialize the variable
        :return: a :py:class:`LpVariable`
        :rtype: :LpVariable
        """
        var = cls(**kwargs)
        var.dj = dj
        var.varValue = varValue
        return var
    from_dict = fromDict

    def add_expression(self, e):
        self.expression = e
        self.addVariableToConstraints(e)

    @classmethod
    def matrix(cls, name, indices=None, lowBound=None, upBound=None, cat=const.LpContinuous, indexStart=[]):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if '%' not in name:
            name += '_%s' * len(indices)
        index = indices[0]
        indices = indices[1:]
        if len(indices) == 0:
            return [LpVariable(name % tuple(indexStart + [i]), lowBound, upBound, cat) for i in index]
        else:
            return [LpVariable.matrix(name, indices, lowBound, upBound, cat, indexStart + [i]) for i in index]

    @classmethod
    def dicts(cls, name, indices=None, lowBound=None, upBound=None, cat=const.LpContinuous, indexStart=[]):
        """
        This function creates a dictionary of :py:class:`LpVariable` with the specified associated parameters.

        :param name: The prefix to the name of each LP variable created
        :param indices: A list of strings of the keys to the dictionary of LP
            variables, and the main part of the variable name itself
        :param lowBound: The lower bound on these variables' range. Default is
            negative infinity
        :param upBound: The upper bound on these variables' range. Default is
            positive infinity
        :param cat: The category these variables are in, Integer or
            Continuous(default)

        :return: A dictionary of :py:class:`LpVariable`
        """
        if not isinstance(indices, tuple):
            indices = (indices,)
        if '%' not in name:
            name += '_%s' * len(indices)
        index = indices[0]
        indices = indices[1:]
        d = {}
        if len(indices) == 0:
            for i in index:
                d[i] = LpVariable(name % tuple(indexStart + [str(i)]), lowBound, upBound, cat)
        else:
            for i in index:
                d[i] = LpVariable.dicts(name, indices, lowBound, upBound, cat, indexStart + [i])
        return d

    @classmethod
    def dict(cls, name, indices, lowBound=None, upBound=None, cat=const.LpContinuous):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if '%' not in name:
            name += '_%s' * len(indices)
        lists = indices
        if len(indices) > 1:
            res = []
            while len(lists):
                first = lists[-1]
                nres = []
                if res:
                    if first:
                        for f in first:
                            nres.extend([[f] + r for r in res])
                    else:
                        nres = res
                    res = nres
                else:
                    res = [[f] for f in first]
                lists = lists[:-1]
            index = [tuple(r) for r in res]
        elif len(indices) == 1:
            index = indices[0]
        else:
            return {}
        d = {}
        for i in index:
            d[i] = cls(name % i, lowBound, upBound, cat)
        return d

    def getLb(self):
        return self.lowBound

    def getUb(self):
        return self.upBound

    def bounds(self, low, up):
        self.lowBound = low
        self.upBound = up
        self.modified = True

    def positive(self):
        self.bounds(0, None)

    def value(self):
        return self.varValue

    def round(self, epsInt=1e-05, eps=1e-07):
        if self.varValue is not None:
            if self.upBound != None and self.varValue > self.upBound and (self.varValue <= self.upBound + eps):
                self.varValue = self.upBound
            elif self.lowBound != None and self.varValue < self.lowBound and (self.varValue >= self.lowBound - eps):
                self.varValue = self.lowBound
            if self.cat == const.LpInteger and abs(round(self.varValue) - self.varValue) <= epsInt:
                self.varValue = round(self.varValue)

    def roundedValue(self, eps=1e-05):
        if self.cat == const.LpInteger and self.varValue != None and (abs(self.varValue - round(self.varValue)) <= eps):
            return round(self.varValue)
        else:
            return self.varValue

    def valueOrDefault(self):
        if self.varValue != None:
            return self.varValue
        elif self.lowBound != None:
            if self.upBound != None:
                if 0 >= self.lowBound and 0 <= self.upBound:
                    return 0
                elif self.lowBound >= 0:
                    return self.lowBound
                else:
                    return self.upBound
            elif 0 >= self.lowBound:
                return 0
            else:
                return self.lowBound
        elif self.upBound != None:
            if 0 <= self.upBound:
                return 0
            else:
                return self.upBound
        else:
            return 0

    def valid(self, eps):
        if self.name == '__dummy' and self.varValue is None:
            return True
        if self.varValue is None:
            return False
        if self.upBound is not None and self.varValue > self.upBound + eps:
            return False
        if self.lowBound is not None and self.varValue < self.lowBound - eps:
            return False
        if self.cat == const.LpInteger and abs(round(self.varValue) - self.varValue) > eps:
            return False
        return True

    def infeasibilityGap(self, mip=1):
        if self.varValue == None:
            raise ValueError('variable value is None')
        if self.upBound != None and self.varValue > self.upBound:
            return self.varValue - self.upBound
        if self.lowBound != None and self.varValue < self.lowBound:
            return self.varValue - self.lowBound
        if mip and self.cat == const.LpInteger and (round(self.varValue) - self.varValue != 0):
            return round(self.varValue) - self.varValue
        return 0

    def isBinary(self):
        return self.cat == const.LpInteger and self.lowBound == 0 and (self.upBound == 1)

    def isInteger(self):
        return self.cat == const.LpInteger

    def isFree(self):
        return self.lowBound is None and self.upBound is None

    def isConstant(self):
        return self.lowBound is not None and self.upBound == self.lowBound

    def isPositive(self):
        return self.lowBound == 0 and self.upBound is None

    def asCplexLpVariable(self):
        if self.isFree():
            return self.name + ' free'
        if self.isConstant():
            return self.name + f' = {self.lowBound:.12g}'
        if self.lowBound == None:
            s = '-inf <= '
        elif self.lowBound == 0 and self.cat == const.LpContinuous:
            s = ''
        else:
            s = f'{self.lowBound:.12g} <= '
        s += self.name
        if self.upBound is not None:
            s += f' <= {self.upBound:.12g}'
        return s

    def asCplexLpAffineExpression(self, name, constant=1):
        return LpAffineExpression(self).asCplexLpAffineExpression(name, constant)

    def __ne__(self, other):
        if isinstance(other, LpElement):
            return self.name is not other.name
        elif isinstance(other, LpAffineExpression):
            if other.isAtomic():
                return self is not other.atom()
            else:
                return 1
        else:
            return 1

    def __bool__(self):
        return bool(self.roundedValue())

    def addVariableToConstraints(self, e):
        """adds a variable to the constraints indicated by
        the LpConstraintVars in e
        """
        for constraint, coeff in e.items():
            constraint.addVariable(self, coeff)

    def setInitialValue(self, val, check=True):
        """
        sets the initial value of the variable to `val`
        May be used for warmStart a solver, if supported by the solver

        :param float val: value to set to variable
        :param bool check: if True, we check if the value fits inside the variable bounds
        :return: True if the value was set
        :raises ValueError: if check=True and the value does not fit inside the bounds
        """
        lb = self.lowBound
        ub = self.upBound
        config = [('smaller', 'lowBound', lb, lambda: val < lb), ('greater', 'upBound', ub, lambda: val > ub)]
        for rel, bound_name, bound_value, condition in config:
            if bound_value is not None and condition():
                if not check:
                    return False
                raise ValueError('In variable {}, initial value {} is {} than {} {}'.format(self.name, val, rel, bound_name, bound_value))
        self.varValue = val
        return True

    def fixValue(self):
        """
        changes lower bound and upper bound to the initial value if exists.
        :return: None
        """
        val = self.varValue
        if val is not None:
            self.bounds(val, val)

    def isFixed(self):
        """

        :return: True if upBound and lowBound are the same
        :rtype: bool
        """
        return self.isConstant()

    def unfixValue(self):
        self.bounds(self._lowbound_original, self._upbound_original)