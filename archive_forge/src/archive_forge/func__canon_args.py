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
def _canon_args(self, expr) -> Tuple[List[Any], List[Any]]:
    """Canonicalize arguments of an expression.

        Like Canonicalization.canonicalize_tree, but preserves signs.
        """
    canon_args = []
    constrs = []
    for arg in expr.args:
        canon_arg, c = self._canonicalize_tree(arg)
        if isinstance(canon_arg, Variable):
            if arg.is_nonneg():
                canon_arg.attributes['nonneg'] = True
            elif arg.is_nonpos():
                canon_arg.attributes['nonpos'] = True
        canon_args += [canon_arg]
        constrs += c
    return (canon_args, constrs)