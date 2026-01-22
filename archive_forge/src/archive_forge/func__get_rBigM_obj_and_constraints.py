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
def _get_rBigM_obj_and_constraints(self, instance_rBigM):
    rBigM_obj = next(instance_rBigM.component_data_objects(Objective, active=True), None)
    if rBigM_obj is None:
        raise GDP_Error('Cannot apply cutting planes transformation without an active objective in the model!')
    fme = TransformationFactory('contrib.fourier_motzkin_elimination')
    rBigM_linear_constraints = []
    for cons in instance_rBigM.component_data_objects(Constraint, descend_into=Block, sort=SortComponents.deterministic, active=True):
        body = cons.body
        if body.polynomial_degree() != 1:
            continue
        rBigM_linear_constraints.extend(fme._process_constraint(cons))
    return (rBigM_obj, rBigM_linear_constraints)