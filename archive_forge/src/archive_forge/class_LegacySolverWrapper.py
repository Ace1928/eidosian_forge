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
class LegacySolverWrapper:
    """
    Class to map the new solver interface features into the legacy solver
    interface. Necessary for backwards compatibility.
    """

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        """Exit statement - enables `with` statements."""

    def _map_config(self, tee, load_solutions, symbolic_solver_labels, timelimit, report_timing, raise_exception_on_nonoptimal_result, solver_io, suffixes, logfile, keepfiles, solnfile, options):
        """Map between legacy and new interface configuration options"""
        self.config = self.config()
        self.config.tee = tee
        self.config.load_solutions = load_solutions
        self.config.symbolic_solver_labels = symbolic_solver_labels
        self.config.time_limit = timelimit
        self.config.solver_options.set_value(options)
        self.config.raise_exception_on_nonoptimal_result = raise_exception_on_nonoptimal_result
        if solver_io is not None:
            raise NotImplementedError('Still working on this')
        if suffixes is not None:
            raise NotImplementedError('Still working on this')
        if logfile is not None:
            raise NotImplementedError('Still working on this')
        if keepfiles or 'keepfiles' in self.config:
            cwd = os.getcwd()
            deprecation_warning(f'`keepfiles` has been deprecated in the new solver interface. Use `working_dir` instead to designate a directory in which files should be generated and saved. Setting `working_dir` to `{cwd}`.', version='6.7.1')
            self.config.working_dir = cwd
        if solnfile is not None:
            if 'filename' in self.config:
                filename = os.path.splitext(solnfile)[0]
                self.config.filename = filename

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

    def solve(self, model: _BlockData, tee: bool=False, load_solutions: bool=True, logfile: Optional[str]=None, solnfile: Optional[str]=None, timelimit: Optional[float]=None, report_timing: bool=False, solver_io: Optional[str]=None, suffixes: Optional[Sequence]=None, options: Optional[Dict]=None, keepfiles: bool=False, symbolic_solver_labels: bool=False, raise_exception_on_nonoptimal_result: bool=False):
        """
        Solve method: maps new solve method style to backwards compatible version.

        Returns
        -------
        legacy_results
            Legacy results object

        """
        original_config = self.config
        self._map_config(tee, load_solutions, symbolic_solver_labels, timelimit, report_timing, raise_exception_on_nonoptimal_result, solver_io, suffixes, logfile, keepfiles, solnfile, options)
        results: Results = super().solve(model)
        legacy_results, legacy_soln = self._map_results(model, results)
        legacy_results = self._solution_handler(load_solutions, model, results, legacy_results, legacy_soln)
        self.config = original_config
        return legacy_results

    def available(self, exception_flag=True):
        """
        Returns a bool determining whether the requested solver is available
        on the system.
        """
        ans = super().available()
        if exception_flag and (not ans):
            raise ApplicationError(f'Solver {self.__class__} is not available ({ans}).')
        return bool(ans)

    def license_is_valid(self) -> bool:
        """Test if the solver license is valid on this system.

        Note that this method is included for compatibility with the
        legacy SolverFactory interface.  Unlicensed or open source
        solvers will return True by definition.  Licensed solvers will
        return True if a valid license is found.

        Returns
        -------
        available: bool
            True if the solver license is valid. Otherwise, False.

        """
        return bool(self.available())