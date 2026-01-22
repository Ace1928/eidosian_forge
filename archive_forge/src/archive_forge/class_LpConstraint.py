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
class LpConstraint(LpAffineExpression):
    """An LP constraint"""

    def __init__(self, e=None, sense=const.LpConstraintEQ, name=None, rhs=None):
        """
        :param e: an instance of :class:`LpAffineExpression`
        :param sense: one of :data:`~pulp.const.LpConstraintEQ`, :data:`~pulp.const.LpConstraintGE`, :data:`~pulp.const.LpConstraintLE` (0, 1, -1 respectively)
        :param name: identifying string
        :param rhs: numerical value of constraint target
        """
        LpAffineExpression.__init__(self, e, name=name)
        if rhs is not None:
            self.constant -= rhs
        self.sense = sense
        self.pi = None
        self.slack = None
        self.modified = True

    def getLb(self):
        if self.sense == const.LpConstraintGE or self.sense == const.LpConstraintEQ:
            return -self.constant
        else:
            return None

    def getUb(self):
        if self.sense == const.LpConstraintLE or self.sense == const.LpConstraintEQ:
            return -self.constant
        else:
            return None

    def __str__(self):
        s = LpAffineExpression.__str__(self, 0)
        if self.sense is not None:
            s += ' ' + const.LpConstraintSenses[self.sense] + ' ' + str(-self.constant)
        return s

    def asCplexLpConstraint(self, name):
        """
        Returns a constraint as a string
        """
        result, line = self.asCplexVariablesOnly(name)
        if not list(self.keys()):
            line += ['0']
        c = -self.constant
        if c == 0:
            c = 0
        term = f' {const.LpConstraintSenses[self.sense]} {c:.12g}'
        if self._count_characters(line) + len(term) > const.LpCplexLPLineSize:
            result += [''.join(line)]
            line = [term]
        else:
            line += [term]
        result += [''.join(line)]
        result = '%s\n' % '\n'.join(result)
        return result

    def changeRHS(self, RHS):
        """
        alters the RHS of a constraint so that it can be modified in a resolve
        """
        self.constant = -RHS
        self.modified = True

    def __repr__(self):
        s = LpAffineExpression.__repr__(self)
        if self.sense is not None:
            s += ' ' + const.LpConstraintSenses[self.sense] + ' 0'
        return s

    def copy(self):
        """Make a copy of self"""
        return LpConstraint(self, self.sense)

    def emptyCopy(self):
        return LpConstraint(sense=self.sense)

    def addInPlace(self, other):
        if isinstance(other, LpConstraint):
            if self.sense * other.sense >= 0:
                LpAffineExpression.addInPlace(self, other)
                self.sense |= other.sense
            else:
                LpAffineExpression.subInPlace(self, other)
                self.sense |= -other.sense
        elif isinstance(other, list):
            for e in other:
                self.addInPlace(e)
        else:
            LpAffineExpression.addInPlace(self, other)
        return self

    def subInPlace(self, other):
        if isinstance(other, LpConstraint):
            if self.sense * other.sense <= 0:
                LpAffineExpression.subInPlace(self, other)
                self.sense |= -other.sense
            else:
                LpAffineExpression.addInPlace(self, other)
                self.sense |= other.sense
        elif isinstance(other, list):
            for e in other:
                self.subInPlace(e)
        else:
            LpAffineExpression.subInPlace(self, other)
        return self

    def __neg__(self):
        c = LpAffineExpression.__neg__(self)
        c.sense = -c.sense
        return c

    def __add__(self, other):
        return self.copy().addInPlace(other)

    def __radd__(self, other):
        return self.copy().addInPlace(other)

    def __sub__(self, other):
        return self.copy().subInPlace(other)

    def __rsub__(self, other):
        return (-self).addInPlace(other)

    def __mul__(self, other):
        if isinstance(other, LpConstraint):
            c = LpAffineExpression.__mul__(self, other)
            if c.sense == 0:
                c.sense = other.sense
            elif other.sense != 0:
                c.sense *= other.sense
            return c
        else:
            return LpAffineExpression.__mul__(self, other)

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other, LpConstraint):
            c = LpAffineExpression.__div__(self, other)
            if c.sense == 0:
                c.sense = other.sense
            elif other.sense != 0:
                c.sense *= other.sense
            return c
        else:
            return LpAffineExpression.__mul__(self, other)

    def __rdiv__(self, other):
        if isinstance(other, LpConstraint):
            c = LpAffineExpression.__rdiv__(self, other)
            if c.sense == 0:
                c.sense = other.sense
            elif other.sense != 0:
                c.sense *= other.sense
            return c
        else:
            return LpAffineExpression.__mul__(self, other)

    def valid(self, eps=0):
        val = self.value()
        if self.sense == const.LpConstraintEQ:
            return abs(val) <= eps
        else:
            return val * self.sense >= -eps

    def makeElasticSubProblem(self, *args, **kwargs):
        """
        Builds an elastic subproblem by adding variables to a hard constraint

        uses FixedElasticSubProblem
        """
        return FixedElasticSubProblem(self, *args, **kwargs)

    def toDict(self):
        """
        exports constraint information into a dictionary

        :return: dictionary with all the constraint information
        """
        return dict(sense=self.sense, pi=self.pi, constant=self.constant, name=self.name, coefficients=LpAffineExpression.toDict(self))

    @classmethod
    def fromDict(cls, _dict):
        """
        Initializes a constraint object from a dictionary with necessary information

        :param dict _dict: dictionary with data
        :return: a new :py:class:`LpConstraint`
        """
        const = cls(e=_dict['coefficients'], rhs=-_dict['constant'], name=_dict['name'], sense=_dict['sense'])
        const.pi = _dict['pi']
        return const
    from_dict = fromDict