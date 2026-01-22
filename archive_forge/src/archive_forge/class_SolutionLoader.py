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
class SolutionLoader(SolutionLoaderBase):

    def __init__(self, primals: Optional[MutableMapping], duals: Optional[MutableMapping], slacks: Optional[MutableMapping], reduced_costs: Optional[MutableMapping]):
        """
        Parameters
        ----------
        primals: dict
            maps id(Var) to (var, value)
        duals: dict
            maps Constraint to dual value
        slacks: dict
            maps Constraint to slack value
        reduced_costs: dict
            maps id(Var) to (var, reduced_cost)
        """
        self._primals = primals
        self._duals = duals
        self._slacks = slacks
        self._reduced_costs = reduced_costs

    def get_primals(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> Mapping[_GeneralVarData, float]:
        if self._primals is None:
            raise RuntimeError('Solution loader does not currently have a valid solution. Please check the termination condition.')
        if vars_to_load is None:
            return ComponentMap(self._primals.values())
        else:
            primals = ComponentMap()
            for v in vars_to_load:
                primals[v] = self._primals[id(v)][1]
            return primals

    def get_duals(self, cons_to_load: Optional[Sequence[_GeneralConstraintData]]=None) -> Dict[_GeneralConstraintData, float]:
        if self._duals is None:
            raise RuntimeError('Solution loader does not currently have valid duals. Please check the termination condition and ensure the solver returns duals for the given problem type.')
        if cons_to_load is None:
            duals = dict(self._duals)
        else:
            duals = dict()
            for c in cons_to_load:
                duals[c] = self._duals[c]
        return duals

    def get_slacks(self, cons_to_load: Optional[Sequence[_GeneralConstraintData]]=None) -> Dict[_GeneralConstraintData, float]:
        if self._slacks is None:
            raise RuntimeError('Solution loader does not currently have valid slacks. Please check the termination condition and ensure the solver returns slacks for the given problem type.')
        if cons_to_load is None:
            slacks = dict(self._slacks)
        else:
            slacks = dict()
            for c in cons_to_load:
                slacks[c] = self._slacks[c]
        return slacks

    def get_reduced_costs(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> Mapping[_GeneralVarData, float]:
        if self._reduced_costs is None:
            raise RuntimeError('Solution loader does not currently have valid reduced costs. Please check the termination condition and ensure the solver returns reduced costs for the given problem type.')
        if vars_to_load is None:
            rc = ComponentMap(self._reduced_costs.values())
        else:
            rc = ComponentMap()
            for v in vars_to_load:
                rc[v] = self._reduced_costs[id(v)][1]
        return rc