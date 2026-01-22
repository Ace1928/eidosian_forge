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
def _get_cone_data(self, con):
    cone_type, cone_param, cone_members = (None, 0, None)
    if isinstance(con, quadratic):
        cone_type = mosek.conetype.quad
        cone_members = [con.r] + list(con.x)
    elif isinstance(con, rotated_quadratic):
        cone_type = mosek.conetype.rquad
        cone_members = [con.r1, con.r2] + list(con.x)
    elif self._version[0] == 9:
        if isinstance(con, primal_exponential):
            cone_type = mosek.conetype.pexp
            cone_members = [con.r, con.x1, con.x2]
        elif isinstance(con, primal_power):
            cone_type = mosek.conetype.ppow
            cone_param = value(con.alpha)
            cone_members = [con.r1, con.r2] + list(con.x)
        elif isinstance(con, dual_exponential):
            cone_type = mosek.conetype.dexp
            cone_members = [con.r, con.x1, con.x2]
        elif isinstance(con, dual_power):
            cone_type = mosek.conetype.dpow
            cone_param = value(con.alpha)
            cone_members = [con.r1, con.r2] + list(con.x)
        else:
            raise UnsupportedDomainError('MOSEK version 9 does not support {}.'.format(type(con)))
    else:
        raise UnsupportedDomainError('MOSEK version {} does not support {}'.format(self._version[0], type(con)))
    return (cone_type, cone_param, ComponentSet(cone_members))