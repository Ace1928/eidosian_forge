import logging
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, Union
import numpy as np
from scipy.sparse import dok_matrix
import cvxpy.settings as s
from cvxpy import Zero
from cvxpy.constraints import SOC, ExpCone, NonNeg
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
def add_model_lin_constr(self, model: ScipModel, variables: List, rows: Iterator, ctype: str, A: dok_matrix, b: np.ndarray) -> List:
    """Adds EQ/LEQ constraints to the model using the data from mat and vec.

        Return list contains constraints.
        """
    from pyscipopt.scip import quicksum
    constraints = []
    expr_list = {i: [] for i in rows}
    for (i, j), c in A.items():
        v = variables[j]
        try:
            expr_list[i].append((c, v))
        except Exception:
            pass
    for i in rows:
        if expr_list[i]:
            expression = quicksum((coeff * var for coeff, var in expr_list[i]))
            constraint = model.addCons(expression == b[i] if ctype == ConstraintTypes.EQUAL else expression <= b[i])
            constraints.append(constraint)
        else:
            constraints.append(None)
    return constraints