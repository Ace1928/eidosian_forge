import abc
from typing import Sequence, Dict, Optional, Mapping, NoReturn
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr import value
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.sol_reader import SolFileData
from pyomo.repn.plugins.nl_writer import NLWriterInfo
from pyomo.core.expr.visitor import replace_expressions
class SolSolutionLoader(SolutionLoaderBase):

    def __init__(self, sol_data: SolFileData, nl_info: NLWriterInfo) -> None:
        self._sol_data = sol_data
        self._nl_info = nl_info

    def load_vars(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> NoReturn:
        if self._nl_info is None:
            raise RuntimeError('Solution loader does not currently have a valid solution. Please check results.TerminationCondition and/or results.SolutionStatus.')
        if self._sol_data is None:
            assert len(self._nl_info.variables) == 0
        elif self._nl_info.scaling:
            for v, val, scale in zip(self._nl_info.variables, self._sol_data.primals, self._nl_info.scaling.variables):
                v.set_value(val / scale, skip_validation=True)
        else:
            for v, val in zip(self._nl_info.variables, self._sol_data.primals):
                v.set_value(val, skip_validation=True)
        for v, v_expr in self._nl_info.eliminated_vars:
            v.value = value(v_expr)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> Mapping[_GeneralVarData, float]:
        if self._nl_info is None:
            raise RuntimeError('Solution loader does not currently have a valid solution. Please check results.TerminationCondition and/or results.SolutionStatus.')
        val_map = dict()
        if self._sol_data is None:
            assert len(self._nl_info.variables) == 0
        else:
            if self._nl_info.scaling is None:
                scale_list = [1] * len(self._nl_info.variables)
            else:
                scale_list = self._nl_info.scaling.variables
            for v, val, scale in zip(self._nl_info.variables, self._sol_data.primals, scale_list):
                val_map[id(v)] = val / scale
        for v, v_expr in self._nl_info.eliminated_vars:
            val = replace_expressions(v_expr, substitution_map=val_map)
            v_id = id(v)
            val_map[v_id] = val
        res = ComponentMap()
        if vars_to_load is None:
            vars_to_load = self._nl_info.variables + [v for v, _ in self._nl_info.eliminated_vars]
        for v in vars_to_load:
            res[v] = val_map[id(v)]
        return res

    def get_duals(self, cons_to_load: Optional[Sequence[_GeneralConstraintData]]=None) -> Dict[_GeneralConstraintData, float]:
        if self._nl_info is None:
            raise RuntimeError('Solution loader does not currently have a valid solution. Please check results.TerminationCondition and/or results.SolutionStatus.')
        if len(self._nl_info.eliminated_vars) > 0:
            raise NotImplementedError('For now, turn presolve off (opt.config.writer_config.linear_presolve=False) to get dual variable values.')
        if self._sol_data is None:
            raise DeveloperError('Solution data is empty. This should not have happened. Report this error to the Pyomo Developers.')
        res = dict()
        if self._nl_info.scaling is None:
            scale_list = [1] * len(self._nl_info.constraints)
            obj_scale = 1
        else:
            scale_list = self._nl_info.scaling.constraints
            obj_scale = self._nl_info.scaling.objectives[0]
        if cons_to_load is None:
            cons_to_load = set(self._nl_info.constraints)
        else:
            cons_to_load = set(cons_to_load)
        for c, val, scale in zip(self._nl_info.constraints, self._sol_data.duals, scale_list):
            if c in cons_to_load:
                res[c] = val * scale / obj_scale
        return res