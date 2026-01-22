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
class LpConstraintVar(LpElement):
    """A Constraint that can be treated as a variable when constructing
    a LpProblem by columns
    """

    def __init__(self, name=None, sense=None, rhs=None, e=None):
        LpElement.__init__(self, name)
        self.constraint = LpConstraint(name=self.name, sense=sense, rhs=rhs, e=e)

    def addVariable(self, var, coeff):
        """
        Adds a variable to the constraint with the
        activity coeff
        """
        self.constraint.addterm(var, coeff)

    def value(self):
        return self.constraint.value()