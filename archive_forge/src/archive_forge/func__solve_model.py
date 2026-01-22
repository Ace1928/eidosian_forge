import logging
import os
import re
import sys
import time
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
import pyomo.core.base.var
def _solve_model(self):
    xprob = self._solver_model
    is_mip = xprob.attributes.mipents > 0 or xprob.attributes.sets > 0
    if xprob.attributes.qelems > 0 or xprob.attributes.qcelems > 0:
        xprob.nlpoptimize('g' if is_mip else '')
        self._get_results = self._get_nlp_results
    elif is_mip:
        xprob.mipoptimize()
        self._get_results = self._get_mip_results
    else:
        xprob.lpoptimize()
        self._get_results = self._get_lp_results
    self._solver_model.postsolve()