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
def _setup_subproblems(self, instance, bigM, tighten_relaxation_callback):
    transBlockName, transBlock = self._add_transformation_block(instance)
    transBlock.all_vars = list((v for v in instance.component_data_objects(Var, descend_into=(Block, Disjunct), sort=SortComponents.deterministic) if not v.is_fixed()))
    nm = self._config.cuts_name
    if nm is None:
        cuts_obj = transBlock.cuts = Constraint(NonNegativeIntegers)
    else:
        if instance.component(nm) is not None:
            raise GDP_Error("cuts_name was specified as '%s', but this is already a component on the instance! Please specify a unique name." % nm)
        instance.add_component(nm, Constraint(NonNegativeIntegers))
        cuts_obj = instance.component(nm)
    bigMRelaxation = TransformationFactory('gdp.bigm')
    hullRelaxation = TransformationFactory('gdp.hull')
    relaxIntegrality = TransformationFactory('core.relax_integer_vars')
    tighter_instance = tighten_relaxation_callback(instance)
    instance_rHull = hullRelaxation.create_using(tighter_instance)
    relaxIntegrality.apply_to(instance_rHull, transform_deactivated_blocks=True)
    bigMRelaxation.apply_to(instance, bigM=bigM)
    relaxIntegrality.apply_to(instance, transform_deactivated_blocks=True)
    transBlock_rHull = instance_rHull.component(transBlockName)
    transBlock_rHull.xstar = Param(range(len(transBlock.all_vars)), mutable=True, default=0, within=Reals)
    extendedSpaceCuts = transBlock_rHull.extendedSpaceCuts = Block()
    extendedSpaceCuts.deactivate()
    extendedSpaceCuts.cuts = Constraint(Any)
    var_info = [(v, transBlock_rHull.all_vars[i], transBlock_rHull.xstar[i]) for i, v in enumerate(transBlock.all_vars)]
    return (instance, cuts_obj, instance_rHull, var_info, transBlockName)