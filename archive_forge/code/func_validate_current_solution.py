from os.path import join, dirname, abspath
import json
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core import Suffix, Var, Constraint, Objective
from pyomo.opt import ProblemFormat, SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
def validate_current_solution(self, **kwds):
    """
        Validate the solution
        """
    assert self.model is not None
    assert self.results_file is not None
    model = self.model
    suffixes = dict(((suffix, getattr(model, suffix)) for suffix in kwds.pop('suffixes', [])))
    exclude = kwds.pop('exclude_suffixes', set())
    for suf in suffixes.values():
        assert suf.import_enabled()
    solution = None
    error_str = 'Difference in solution for {0}.{1}:\n\tBaseline - {2}\n\tCurrent - {3}'
    with open(self.results_file, 'r') as f:
        try:
            solution = json.load(f)
        except:
            return (False, 'Problem reading file ' + self.results_file)
    for var in model.component_data_objects(Var):
        var_value_sol = solution[var.name]['value']
        var_value = var.value
        if not (var_value is None and var_value_sol is None):
            if (var_value is None) ^ (var_value_sol is None) or abs(var_value_sol - var_value) > self.diff_tol:
                return (False, error_str.format(var.name, 'value', var_value_sol, var_value))
        if not solution[var.name]['stale'] is var.stale:
            return (False, error_str.format(var.name, 'stale', solution[var.name]['stale'], var.stale))
        for suffix_name, suffix in suffixes.items():
            _ex = exclude.get(suffix_name, None)
            if suffix_name in solution[var.name]:
                if suffix.get(var) is None:
                    if _ex is not None and (not _ex[1] or var.name in _ex[1]):
                        continue
                    if not solution[var.name][suffix_name] in solution['suffix defaults'][suffix_name]:
                        return (False, error_str.format(var.name, suffix, solution[var.name][suffix_name], 'none defined'))
                elif _ex is not None and _ex[0] and (not _ex[1] or var.name in _ex[1]):
                    return (False, 'Expected solution to be missing suffix %s' % suffix_name)
                elif not abs(solution[var.name][suffix_name] - suffix.get(var)) < self.diff_tol:
                    return (False, error_str.format(var.name, suffix, solution[var.name][suffix_name], suffix.get(var)))
    for con in model.component_data_objects(Constraint):
        con_value_sol = solution[con.name]['value']
        con_value = con(exception=False)
        if not (con_value is None and con_value_sol is None):
            if (con_value is None) ^ (con_value_sol is None) or abs(con_value_sol - con_value) > self.diff_tol:
                return (False, error_str.format(con.name, 'value', con_value_sol, con_value))
        for suffix_name, suffix in suffixes.items():
            _ex = exclude.get(suffix_name, None)
            if suffix_name in solution[con.name]:
                if suffix.get(con) is None:
                    if _ex is not None and (not _ex[1] or con.name in _ex[1]):
                        continue
                    if not solution[con.name][suffix_name] in solution['suffix defaults'][suffix_name]:
                        return (False, error_str.format(con.name, suffix, solution[con.name][suffix_name], 'none defined'))
                elif _ex is not None and _ex[0] and (not _ex[1] or con.name in _ex[1]):
                    return (False, 'Expected solution to be missing suffix %s' % suffix_name)
                elif not abs(solution[con.name][suffix_name] - suffix.get(con)) < self.diff_tol:
                    return (False, error_str.format(con.name, suffix, solution[con.name][suffix_name], suffix.get(con)))
    for obj in model.component_data_objects(Objective):
        obj_value_sol = solution[obj.name]['value']
        obj_value = obj(exception=False)
        if not (obj_value is None and obj_value_sol is None):
            if (obj_value is None) ^ (obj_value_sol is None) or abs(obj_value_sol - obj_value) > self.diff_tol:
                return (False, error_str.format(obj.name, 'value', obj_value_sol, obj_value))
        for suffix_name, suffix in suffixes.items():
            _ex = exclude.get(suffix_name, None)
            if suffix_name in solution[obj.name]:
                if suffix.get(obj) is None:
                    if _ex is not None and (not _ex[1] or obj.name in _ex[1]):
                        continue
                    if not solution[obj.name][suffix_name] in solution['suffix defaults'][suffix_name]:
                        return (False, error_str.format(obj.name, suffix, solution[obj.name][suffix_name], 'none defined'))
                elif _ex is not None and _ex[0] and (not _ex[1] or obj.name in _ex[1]):
                    return (False, 'Expected solution to be missing suffix %s' % suffix_name)
                elif not abs(solution[obj.name][suffix_name] - suffix.get(obj)) < self.diff_tol:
                    return (False, error_str.format(obj.name, suffix, solution[obj.name][suffix_name], suffix.get(obj)))
    first = True
    for block in model.block_data_objects():
        if first:
            first = False
            continue
        for suffix_name, suffix in suffixes.items():
            _ex = exclude.get(suffix_name, None)
            if solution[block.name] is not None and suffix_name in solution[block.name]:
                if suffix.get(block) is None:
                    if _ex is not None and (not _ex[1] or block.name in _ex[1]):
                        continue
                    if not solution[block.name][suffix_name] in solution['suffix defaults'][suffix_name]:
                        return (False, error_str.format(block.name, suffix, solution[block.name][suffix_name], 'none defined'))
                elif _ex is not None and _ex[0] and (not _ex[1] or block.name in _ex[1]):
                    return (False, 'Expected solution to be missing suffix %s' % suffix_name)
                elif not abs(solution[block.name][suffix_name] - suffix.get(block)) < self.diff_tol:
                    return (False, error_str.format(block.name, suffix, solution[block.name][suffix_name], suffix.get(block)))
    return (True, '')