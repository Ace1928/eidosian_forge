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
def _get_acc_domain(self, cone):
    domidx, domdim, members = (None, 0, None)
    if isinstance(cone, quadratic):
        domdim = 1 + len(cone.x)
        domidx = self._solver_model.appendquadraticconedomain(domdim)
        members = [cone.r] + list(cone.x)
    elif isinstance(cone, rotated_quadratic):
        domdim = 2 + len(cone.x)
        domidx = self._solver_model.appendrquadraticconedomain(domdim)
        members = [cone.r1, cone.r2] + list(cone.x)
    elif isinstance(cone, primal_exponential):
        domdim = 3
        domidx = self._solver_model.appendprimalexpconedomain()
        members = [cone.r, cone.x1, cone.x2]
    elif isinstance(cone, dual_exponential):
        domdim = 3
        domidx = self._solver_model.appenddualexpconedomain()
        members = [cone.r, cone.x1, cone.x2]
    elif isinstance(cone, primal_power):
        domdim = 2 + len(cone.x)
        domidx = self._solver_model.appendprimalpowerconedomain(domdim, [value(cone.alpha), 1 - value(cone.alpha)])
        members = [cone.r1, cone.r2] + list(cone.x)
    elif isinstance(cone, dual_power):
        domdim = 2 + len(cone.x)
        domidx = self._solver_model.appenddualpowerconedomain(domdim, [value(cone.alpha), 1 - value(cone.alpha)])
        members = [cone.r1, cone.r2] + list(cone.x)
    elif isinstance(cone, primal_geomean):
        domdim = len(cone.r) + 1
        domidx = self._solver_model.appendprimalgeomeanconedomain(domdim)
        members = list(cone.r) + [cone.x]
    elif isinstance(cone, dual_geomean):
        domdim = len(cone.r) + 1
        domidx = self._solver_model.appenddualgeomeanconedomain(domdim)
        members = list(cone.r) + [cone.x]
    elif isinstance(cone, svec_psdcone):
        domdim = len(cone.x)
        domidx = self._solver_model.appendsvecpsdconedomain(domdim)
        members = list(cone.x)
    return (domdim, domidx, members)