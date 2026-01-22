import abc
import enum
from typing import Sequence, Dict, Optional, Mapping, NoReturn, List, Tuple
import os
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.errors import ApplicationError
from pyomo.common.deprecation import deprecation_warning
from pyomo.opt.results.results_ import SolverResults as LegacySolverResults
from pyomo.opt.results.solution import Solution as LegacySolution
from pyomo.core.kernel.objective import minimize
from pyomo.core.base import SymbolMap
from pyomo.core.base.label import NumericLabeler
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.config import SolverConfig, PersistentSolverConfig
from pyomo.contrib.solver.util import get_objective
from pyomo.contrib.solver.results import (
def _map_results(self, model, results):
    """Map between legacy and new Results objects"""
    legacy_results = LegacySolverResults()
    legacy_soln = LegacySolution()
    legacy_results.solver.status = legacy_solver_status_map[results.termination_condition]
    legacy_results.solver.termination_condition = legacy_termination_condition_map[results.termination_condition]
    legacy_soln.status = legacy_solution_status_map[results.solution_status]
    legacy_results.solver.termination_message = str(results.termination_condition)
    obj = get_objective(model)
    if len(list(obj)) > 0:
        legacy_results.problem.sense = obj.sense
        if obj.sense == minimize:
            legacy_results.problem.lower_bound = results.objective_bound
            legacy_results.problem.upper_bound = results.incumbent_objective
        else:
            legacy_results.problem.upper_bound = results.objective_bound
            legacy_results.problem.lower_bound = results.incumbent_objective
    if results.incumbent_objective is not None and results.objective_bound is not None:
        legacy_soln.gap = abs(results.incumbent_objective - results.objective_bound)
    else:
        legacy_soln.gap = None
    return (legacy_results, legacy_soln)