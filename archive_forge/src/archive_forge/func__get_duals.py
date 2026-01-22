from collections.abc import Iterable
import logging
import math
from typing import List, Optional
from pyomo.common.collections import ComponentSet, ComponentMap, OrderedSet
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.config import ConfigValue
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.param import _ParamData
from pyomo.core.expr.numvalue import value, is_constant, is_fixed, native_numeric_types
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.solver.base import PersistentSolverBase
from pyomo.contrib.solver.results import Results, TerminationCondition, SolutionStatus
from pyomo.contrib.solver.config import PersistentBranchAndBoundConfig
from pyomo.contrib.solver.persistent import PersistentSolverUtils
from pyomo.contrib.solver.solution import PersistentSolutionLoader
from pyomo.core.staleflag import StaleFlagManager
import sys
import datetime
import io
def _get_duals(self, cons_to_load=None):
    if self._needs_updated:
        self._update_gurobi_model()
    if self._solver_model.Status != gurobipy.GRB.OPTIMAL:
        raise RuntimeError('Solver does not currently have valid duals. Please check the termination condition.')
    con_map = self._pyomo_con_to_solver_con_map
    reverse_con_map = self._solver_con_to_pyomo_con_map
    dual = dict()
    if cons_to_load is None:
        linear_cons_to_load = self._solver_model.getConstrs()
        quadratic_cons_to_load = self._solver_model.getQConstrs()
    else:
        gurobi_cons_to_load = OrderedSet([con_map[pyomo_con] for pyomo_con in cons_to_load])
        linear_cons_to_load = list(gurobi_cons_to_load.intersection(OrderedSet(self._solver_model.getConstrs())))
        quadratic_cons_to_load = list(gurobi_cons_to_load.intersection(OrderedSet(self._solver_model.getQConstrs())))
    linear_vals = self._solver_model.getAttr('Pi', linear_cons_to_load)
    quadratic_vals = self._solver_model.getAttr('QCPi', quadratic_cons_to_load)
    for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
        pyomo_con = reverse_con_map[id(gurobi_con)]
        dual[pyomo_con] = val
    for gurobi_con, val in zip(quadratic_cons_to_load, quadratic_vals):
        pyomo_con = reverse_con_map[id(gurobi_con)]
        dual[pyomo_con] = val
    return dual