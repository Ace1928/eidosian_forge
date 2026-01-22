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
class PersistentSolutionLoader(SolutionLoaderBase):

    def __init__(self, solver):
        self._solver = solver
        self._valid = True

    def _assert_solution_still_valid(self):
        if not self._valid:
            raise RuntimeError('The results in the solver are no longer valid.')

    def get_primals(self, vars_to_load=None):
        self._assert_solution_still_valid()
        return self._solver._get_primals(vars_to_load=vars_to_load)

    def get_duals(self, cons_to_load: Optional[Sequence[_GeneralConstraintData]]=None) -> Dict[_GeneralConstraintData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_duals(cons_to_load=cons_to_load)

    def get_reduced_costs(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> Mapping[_GeneralVarData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_reduced_costs(vars_to_load=vars_to_load)

    def invalidate(self):
        self._valid = False