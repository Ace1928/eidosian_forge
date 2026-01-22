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
def _transform_disjunct(self, disjunct, partition, transBlock):
    if not disjunct.active:
        if disjunct.indicator_var.is_fixed():
            if not value(disjunct.indicator_var):
                return
            else:
                raise GDP_Error("The disjunct '%s' is deactivated, but the indicator_var is fixed to %s. This makes no sense." % (disjunct.name, value(disjunct.indicator_var)))
        if disjunct._transformation_block is None:
            raise GDP_Error("The disjunct '%s' is deactivated, but the indicator_var is not fixed and the disjunct does not appear to have been relaxed. This makes no sense. (If the intent is to deactivate the disjunct, fix its indicator_var to False.)" % (disjunct.name,))
    if disjunct._transformation_block is not None:
        raise GDP_Error("The disjunct '%s' has been transformed, but a disjunction it appears in has not. Putting the same disjunct in multiple disjunctions is not supported." % disjunct.name)
    transformed_disjunct = Disjunct()
    disjunct._transformation_block = weakref_ref(transformed_disjunct)
    transBlock.add_component(unique_component_name(transBlock, disjunct.getname(fully_qualified=True)), transformed_disjunct)
    if disjunct.indicator_var.fixed:
        transformed_disjunct.indicator_var.fix(value(disjunct.indicator_var))
    for disjunction in disjunct.component_data_objects(Disjunction, active=True, sort=SortComponents.deterministic, descend_into=Block):
        self._transform_disjunctionData(disjunction, disjunction.index(), None, transformed_disjunct)
    for var in disjunct.component_objects(Var, descend_into=Block, active=None):
        transformed_disjunct.add_component(unique_component_name(transformed_disjunct, var.getname(fully_qualified=True)), Reference(var))
    logical_constraints = LogicalConstraintList()
    transformed_disjunct.add_component(unique_component_name(transformed_disjunct, 'logical_constraints'), logical_constraints)
    for cons in disjunct.component_data_objects(LogicalConstraint, descend_into=Block, active=None):
        logical_constraints.add(cons.expr)
        cons.deactivate()
    for obj in disjunct.component_data_objects(active=True, sort=SortComponents.deterministic, descend_into=Block):
        handler = self.handlers.get(obj.ctype, None)
        if not handler:
            if handler is None:
                raise GDP_Error('No partition_disjuncts transformation handler registered for modeling components of type %s. If your disjuncts contain non-GDP Pyomo components that require transformation, please transform them first.' % obj.ctype)
            continue
        handler(obj, disjunct, transformed_disjunct, transBlock, partition)
    disjunct._deactivate_without_fixing_indicator()
    return transformed_disjunct