from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base.external import ExternalFunction
from pyomo.network import Port
from pyomo.common.collections import ComponentSet
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
from pyomo.opt import SolverFactory
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from weakref import ref as weakref_ref
from math import floor
import logging
def _transform_blockData(self, obj):
    to_transform = []
    for disjunction in obj.component_data_objects(Disjunction, active=True, sort=SortComponents.deterministic, descend_into=Block):
        to_transform.append(disjunction)
    for disjunction in to_transform:
        self._transform_disjunctionData(disjunction, disjunction.index())