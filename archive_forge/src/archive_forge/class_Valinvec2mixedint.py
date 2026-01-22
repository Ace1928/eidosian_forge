from typing import Callable, List, Tuple
import numpy as np
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
class Valinvec2mixedint(Canonicalization):

    def accepts(self, problem) -> bool:
        return any(FiniteSet in {type(c) for c in problem.constraints})
    CANON_METHODS = {FiniteSet: finite_set_canon}

    def __init__(self, problem=None) -> None:
        super(Valinvec2mixedint, self).__init__(problem=problem, canon_methods=Valinvec2mixedint.CANON_METHODS)