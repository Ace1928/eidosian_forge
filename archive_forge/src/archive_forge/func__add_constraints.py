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
def _add_constraints(self, model: ScipModel, variables: List, A: dok_matrix, b: np.ndarray, dims: Dict[str, Union[int, List]]) -> List:
    """Create a list of constraints."""
    equal_constraints = self.add_model_lin_constr(model=model, variables=variables, rows=range(dims[s.EQ_DIM]), ctype=ConstraintTypes.EQUAL, A=A, b=b)
    leq_start = dims[s.EQ_DIM]
    leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
    inequal_constraints = self.add_model_lin_constr(model=model, variables=variables, rows=range(leq_start, leq_end), ctype=ConstraintTypes.LESS_THAN_OR_EQUAL, A=A, b=b)
    soc_start = leq_end
    soc_constrs = []
    new_leq_constrs = []
    for constr_len in dims[s.SOC_DIM]:
        soc_end = soc_start + constr_len
        soc_constr, new_leq, new_vars = self.add_model_soc_constr(model=model, variables=variables, rows=range(soc_start, soc_end), A=A, b=b)
        soc_constrs.append(soc_constr)
        new_leq_constrs += new_leq
        variables += new_vars
        soc_start += constr_len
    return equal_constraints + inequal_constraints + new_leq_constrs + soc_constrs