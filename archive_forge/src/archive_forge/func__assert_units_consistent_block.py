import logging
from pyomo.core.base.units_container import units, UnitsError
from pyomo.core.base import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.network import Port, Arc
from pyomo.mpec import Complementarity
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.numvalue import native_types
from pyomo.util.components import iter_component
from pyomo.common.collections import ComponentSet
def _assert_units_consistent_block(obj):
    """
    This method gets all the components from the block
    and checks if the units are consistent on each of them
    """
    for component in obj.component_objects(descend_into=False, active=True):
        assert_units_consistent(component)