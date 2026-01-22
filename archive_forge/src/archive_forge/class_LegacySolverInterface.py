import abc
import enum
from typing import (
from pyomo.core.base.constraint import _GeneralConstraintData, Constraint
from pyomo.core.base.sos import _SOSConstraintData, SOSConstraint
from pyomo.core.base.var import _GeneralVarData, Var
from pyomo.core.base.param import _ParamData, Param
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.collections import ComponentMap
from .utils.get_objective import get_objective
from .utils.collect_vars_and_named_exprs import collect_vars_and_named_exprs
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeFloat
from pyomo.common.errors import ApplicationError
from pyomo.opt.base import SolverFactory as LegacySolverFactory
from pyomo.common.factory import Factory
import os
from pyomo.opt.results.results_ import SolverResults as LegacySolverResults
from pyomo.opt.results.solution import (
from pyomo.opt.results.solver import (
from pyomo.core.kernel.objective import minimize
from pyomo.core.base import SymbolMap
import weakref
from .cmodel import cmodel, cmodel_available
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import NumericConstant
class LegacySolverInterface(object):

    def solve(self, model: _BlockData, tee: bool=False, load_solutions: bool=True, logfile: Optional[str]=None, solnfile: Optional[str]=None, timelimit: Optional[float]=None, report_timing: bool=False, solver_io: Optional[str]=None, suffixes: Optional[Sequence]=None, options: Optional[Dict]=None, keepfiles: bool=False, symbolic_solver_labels: bool=False):
        original_config = self.config
        self.config = self.config()
        self.config.stream_solver = tee
        self.config.load_solution = load_solutions
        self.config.symbolic_solver_labels = symbolic_solver_labels
        self.config.time_limit = timelimit
        self.config.report_timing = report_timing
        if solver_io is not None:
            raise NotImplementedError('Still working on this')
        if suffixes is not None:
            raise NotImplementedError('Still working on this')
        if logfile is not None:
            raise NotImplementedError('Still working on this')
        if 'keepfiles' in self.config:
            self.config.keepfiles = keepfiles
        if solnfile is not None:
            if 'filename' in self.config:
                filename = os.path.splitext(solnfile)[0]
                self.config.filename = filename
        original_options = self.options
        if options is not None:
            self.options = options
        results: Results = super(LegacySolverInterface, self).solve(model)
        legacy_results = LegacySolverResults()
        legacy_soln = LegacySolution()
        legacy_results.solver.status = legacy_solver_status_map[results.termination_condition]
        legacy_results.solver.termination_condition = legacy_termination_condition_map[results.termination_condition]
        legacy_soln.status = legacy_solution_status_map[results.termination_condition]
        legacy_results.solver.termination_message = str(results.termination_condition)
        obj = get_objective(model)
        legacy_results.problem.sense = obj.sense
        if obj.sense == minimize:
            legacy_results.problem.lower_bound = results.best_objective_bound
            legacy_results.problem.upper_bound = results.best_feasible_objective
        else:
            legacy_results.problem.upper_bound = results.best_objective_bound
            legacy_results.problem.lower_bound = results.best_feasible_objective
        if results.best_feasible_objective is not None and results.best_objective_bound is not None:
            legacy_soln.gap = abs(results.best_feasible_objective - results.best_objective_bound)
        else:
            legacy_soln.gap = None
        symbol_map = SymbolMap()
        symbol_map.byObject = dict(self.symbol_map.byObject)
        symbol_map.bySymbol = dict(self.symbol_map.bySymbol)
        symbol_map.aliases = dict(self.symbol_map.aliases)
        symbol_map.default_labeler = self.symbol_map.default_labeler
        model.solutions.add_symbol_map(symbol_map)
        legacy_results._smap_id = id(symbol_map)
        delete_legacy_soln = True
        if load_solutions:
            if hasattr(model, 'dual') and model.dual.import_enabled():
                for c, val in results.solution_loader.get_duals().items():
                    model.dual[c] = val
            if hasattr(model, 'slack') and model.slack.import_enabled():
                for c, val in results.solution_loader.get_slacks().items():
                    model.slack[c] = val
            if hasattr(model, 'rc') and model.rc.import_enabled():
                for v, val in results.solution_loader.get_reduced_costs().items():
                    model.rc[v] = val
        elif results.best_feasible_objective is not None:
            delete_legacy_soln = False
            for v, val in results.solution_loader.get_primals().items():
                legacy_soln.variable[symbol_map.getSymbol(v)] = {'Value': val}
            if hasattr(model, 'dual') and model.dual.import_enabled():
                for c, val in results.solution_loader.get_duals().items():
                    legacy_soln.constraint[symbol_map.getSymbol(c)] = {'Dual': val}
            if hasattr(model, 'slack') and model.slack.import_enabled():
                for c, val in results.solution_loader.get_slacks().items():
                    symbol = symbol_map.getSymbol(c)
                    if symbol in legacy_soln.constraint:
                        legacy_soln.constraint[symbol]['Slack'] = val
            if hasattr(model, 'rc') and model.rc.import_enabled():
                for v, val in results.solution_loader.get_reduced_costs().items():
                    legacy_soln.variable['Rc'] = val
        legacy_results.solution.insert(legacy_soln)
        if delete_legacy_soln:
            legacy_results.solution.delete(0)
        self.config = original_config
        self.options = original_options
        return legacy_results

    def available(self, exception_flag=True):
        ans = super(LegacySolverInterface, self).available()
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

    @property
    def options(self):
        for solver_name in ['gurobi', 'ipopt', 'cplex', 'cbc', 'highs']:
            if hasattr(self, solver_name + '_options'):
                return getattr(self, solver_name + '_options')
        raise NotImplementedError('Could not find the correct options')

    @options.setter
    def options(self, val):
        found = False
        for solver_name in ['gurobi', 'ipopt', 'cplex', 'cbc', 'highs']:
            if hasattr(self, solver_name + '_options'):
                setattr(self, solver_name + '_options', val)
                found = True
        if not found:
            raise NotImplementedError('Could not find the correct options')

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass