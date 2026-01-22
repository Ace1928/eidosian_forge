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
def _load_slacks(self, cons_to_load=None):
    if not hasattr(self._pyomo_model, 'slack'):
        self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)
    con_map = self._pyomo_con_to_solver_con_map
    reverse_con_map = self._solver_con_to_pyomo_con_map
    slack = self._pyomo_model.slack
    if cons_to_load is None:
        mosek_cons_to_load = range(self._solver_model.getnumcon())
    else:
        mosek_cons_to_load = set([con_map[pyomo_con] for pyomo_con in cons_to_load])
    Ax = [0] * len(mosek_cons_to_load)
    self._solver_model.getxc(self._whichsol, Ax)
    for con in mosek_cons_to_load:
        pyomo_con = reverse_con_map[con]
        Us = Ls = 0
        bk, lb, ub = self._solver_model.getconbound(con)
        if bk in [mosek.boundkey.fx, mosek.boundkey.ra, mosek.boundkey.up]:
            Us = ub - Ax[con]
        if bk in [mosek.boundkey.fx, mosek.boundkey.ra, mosek.boundkey.lo]:
            Ls = Ax[con] - lb
        if Us > Ls:
            slack[pyomo_con] = Us
        else:
            slack[pyomo_con] = -Ls