from collections import namedtuple
from typing import Any, List, Tuple
import numpy as np
from cvxpy import problems
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.max import max as max_atom
from cvxpy.atoms.min import min as min_atom
from cvxpy.constraints import Inequality
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS
from cvxpy.reductions.dqcp2dcp import inverse, sets, tighten
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solution import Solution
def _canonicalize_tree(self, expr):
    canon_args, constrs = self._canon_args(expr)
    canon_expr, c = self.canonicalize_expr(expr, canon_args)
    constrs += c
    return (canon_expr, constrs)