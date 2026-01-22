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
def _get_lp_results(self, results, soln):
    """Sets up `results` and `soln` and returns whether there is a solution
        to query.
        Returns `True` if a feasible solution is available, `False` otherwise.
        """
    xprob = self._solver_model
    xp = xpress
    xprob_attrs = xprob.attributes
    status = xprob_attrs.lpstatus
    if status == xp.lp_unstarted:
        results.solver.status = SolverStatus.aborted
        results.solver.termination_message = 'Model is not loaded; no solution information is available.'
        results.solver.termination_condition = TerminationCondition.error
        soln.status = SolutionStatus.unknown
    elif status == xp.lp_optimal:
        results.solver.status = SolverStatus.ok
        results.solver.termination_message = 'Model was solved to optimality (subject to tolerances), and an optimal solution is available.'
        results.solver.termination_condition = TerminationCondition.optimal
        soln.status = SolutionStatus.optimal
    elif status == xp.lp_infeas:
        results.solver.status = SolverStatus.warning
        results.solver.termination_message = 'Model was proven to be infeasible'
        results.solver.termination_condition = TerminationCondition.infeasible
        soln.status = SolutionStatus.infeasible
    elif status == xp.lp_cutoff:
        results.solver.status = SolverStatus.ok
        results.solver.termination_message = 'Optimal objective for model was proven to be worse than the cutoff value specified; a solution is available.'
        results.solver.termination_condition = TerminationCondition.minFunctionValue
        soln.status = SolutionStatus.optimal
    elif status == xp.lp_unfinished:
        results.solver.status = SolverStatus.aborted
        results.solver.termination_message = 'Optimization was terminated by the user.'
        results.solver.termination_condition = TerminationCondition.error
        soln.status = SolutionStatus.error
    elif status == xp.lp_unbounded:
        results.solver.status = SolverStatus.warning
        results.solver.termination_message = 'Model was proven to be unbounded.'
        results.solver.termination_condition = TerminationCondition.unbounded
        soln.status = SolutionStatus.unbounded
    elif status == xp.lp_cutoff_in_dual:
        results.solver.status = SolverStatus.ok
        results.solver.termination_message = 'Xpress reported the LP was cutoff in the dual.'
        results.solver.termination_condition = TerminationCondition.minFunctionValue
        soln.status = SolutionStatus.optimal
    elif status == xp.lp_unsolved:
        results.solver.status = SolverStatus.error
        results.solver.termination_message = 'Optimization was terminated due to unrecoverable numerical difficulties.'
        results.solver.termination_condition = TerminationCondition.error
        soln.status = SolutionStatus.error
    elif status == xp.lp_nonconvex:
        results.solver.status = SolverStatus.error
        results.solver.termination_message = 'Optimization was terminated because nonconvex quadratic data were found.'
        results.solver.termination_condition = TerminationCondition.error
        soln.status = SolutionStatus.error
    else:
        results.solver.status = SolverStatus.error
        results.solver.termination_message = 'Unhandled Xpress solve status (' + str(status) + ')'
        results.solver.termination_condition = TerminationCondition.error
        soln.status = SolutionStatus.error
    results.problem.upper_bound = None
    results.problem.lower_bound = None
    try:
        results.problem.upper_bound = xprob_attrs.lpobjval
        results.problem.lower_bound = xprob_attrs.lpobjval
    except (XpressDirect.XpressException, AttributeError):
        pass
    return xprob_attrs.lpstatus in [xp.lp_optimal, xp.lp_cutoff, xp.lp_cutoff_in_dual]