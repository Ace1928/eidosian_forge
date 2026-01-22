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
def _solution_handler(self, load_solutions, model, results, legacy_results, legacy_soln):
    """Method to handle the preferred action for the solution"""
    symbol_map = SymbolMap()
    symbol_map.default_labeler = NumericLabeler('x')
    model.solutions.add_symbol_map(symbol_map)
    legacy_results._smap_id = id(symbol_map)
    delete_legacy_soln = True
    if load_solutions:
        if hasattr(model, 'dual') and model.dual.import_enabled():
            for c, val in results.solution_loader.get_duals().items():
                model.dual[c] = val
        if hasattr(model, 'rc') and model.rc.import_enabled():
            for v, val in results.solution_loader.get_reduced_costs().items():
                model.rc[v] = val
    elif results.incumbent_objective is not None:
        delete_legacy_soln = False
        for v, val in results.solution_loader.get_primals().items():
            legacy_soln.variable[symbol_map.getSymbol(v)] = {'Value': val}
        if hasattr(model, 'dual') and model.dual.import_enabled():
            for c, val in results.solution_loader.get_duals().items():
                legacy_soln.constraint[symbol_map.getSymbol(c)] = {'Dual': val}
        if hasattr(model, 'rc') and model.rc.import_enabled():
            for v, val in results.solution_loader.get_reduced_costs().items():
                legacy_soln.variable['Rc'] = val
    legacy_results.solution.insert(legacy_soln)
    legacy_results.timing_info = results.timing_info
    if delete_legacy_soln:
        legacy_results.solution.delete(0)
    return legacy_results