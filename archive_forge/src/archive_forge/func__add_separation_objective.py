from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def _add_separation_objective(self, var_info, transBlock_rHull):
    for o in transBlock_rHull.model().component_data_objects(Objective):
        o.deactivate()
    norm = self._config.norm
    to_delete = []
    if norm == 2:
        obj_expr = 0
        for i, (x_rbigm, x_hull, x_star) in enumerate(var_info):
            if not x_rbigm.stale:
                obj_expr += (x_hull - x_star) ** 2
            else:
                if self.verbose:
                    logger.info('The variable %s will not be included in the separation problem: It was stale in the rBigM solve.' % x_rbigm.getname(fully_qualified=True))
                to_delete.append(i)
    elif norm == float('inf'):
        u = transBlock_rHull.u = Var(domain=NonNegativeReals)
        inf_cons = transBlock_rHull.inf_norm_linearization = Constraint(NonNegativeIntegers)
        i = 0
        for j, (x_rbigm, x_hull, x_star) in enumerate(var_info):
            if not x_rbigm.stale:
                inf_cons[i] = u - x_hull >= -x_star
                inf_cons[i + 1] = u + x_hull >= x_star
                i += 2
            else:
                if self.verbose:
                    logger.info('The variable %s will not be included in the separation problem: It was stale in the rBigM solve.' % x_rbigm.getname(fully_qualified=True))
                to_delete.append(j)
        self._add_dual_suffix(transBlock_rHull.model())
        obj_expr = u
    for i in sorted(to_delete, reverse=True):
        del var_info[i]
    transBlock_rHull.separation_objective = Objective(expr=obj_expr)