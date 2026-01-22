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
def addConstraint(self, constraint, name=None):
    if not isinstance(constraint, LpConstraint):
        raise TypeError('Can only add LpConstraint objects')
    if name:
        constraint.name = name
    try:
        if constraint.name:
            name = constraint.name
        else:
            name = self.unusedConstraintName()
    except AttributeError:
        raise TypeError('Can only add LpConstraint objects')
    if name in self.constraints:
        if self.noOverlap:
            raise const.PulpError('overlapping constraint names: ' + name)
        else:
            print('Warning: overlapping constraint names:', name)
    self.constraints[name] = constraint
    self.modifiedConstraints.append(constraint)
    self.addVariables(list(constraint.keys()))