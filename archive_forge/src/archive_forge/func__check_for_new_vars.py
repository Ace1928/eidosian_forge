import abc
from typing import List
from pyomo.core.base.constraint import _GeneralConstraintData, Constraint
from pyomo.core.base.sos import _SOSConstraintData, SOSConstraint
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData, Param
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.collections import ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.contrib.solver.util import collect_vars_and_named_exprs, get_objective
def _check_for_new_vars(self, variables: List[_GeneralVarData]):
    new_vars = {}
    for v in variables:
        v_id = id(v)
        if v_id not in self._referenced_variables:
            new_vars[v_id] = v
    self.add_variables(list(new_vars.values()))