import logging
import re
import sys
import itertools
import operator
import pyomo.core.base.var
import pyomo.core.base.constraint
from pyomo.common.dependencies import attempt_import
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import is_fixed, value, minimize, maximize
from pyomo.core.base.suffix import Suffix
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt.base.solvers import OptSolver
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.common.collections import ComponentMap, ComponentSet, Bunch
from pyomo.opt import SolverFactory
from pyomo.core.kernel.conic import (
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
    degree = repn.polynomial_degree()
    if degree is None or degree > max_degree:
        raise DegreeError('MOSEK does not support expressions of degree {}.'.format(degree))
    referenced_vars = ComponentSet(repn.linear_vars)
    indices = tuple((self._pyomo_var_to_solver_var_map[i] for i in repn.linear_vars))
    mosek_arow = (indices, tuple(repn.linear_coefs), repn.constant)
    if len(repn.quadratic_vars) == 0:
        mosek_qexp = ((), (), ())
        return (mosek_arow, mosek_qexp, referenced_vars)
    else:
        q_vars = itertools.chain.from_iterable(repn.quadratic_vars)
        referenced_vars.update(q_vars)
        qsubi, qsubj = zip(*[(i, j) if self._pyomo_var_to_solver_var_map[i] >= self._pyomo_var_to_solver_var_map[j] else (j, i) for i, j in repn.quadratic_vars])
        qsubi = tuple((self._pyomo_var_to_solver_var_map[i] for i in qsubi))
        qsubj = tuple((self._pyomo_var_to_solver_var_map[j] for j in qsubj))
        qvals = tuple((v * 2 if qsubi[i] is qsubj[i] else v for i, v in enumerate(repn.quadratic_coefs)))
        mosek_qexp = (qsubi, qsubj, qvals)
    return (mosek_arow, mosek_qexp, referenced_vars)