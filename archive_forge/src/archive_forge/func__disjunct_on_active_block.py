import logging
from pyomo.core.base import Transformation, Block, Constraint
from pyomo.gdp import Disjunct, GDP_Error, Disjunction
from pyomo.core import TraversalStrategy, TransformationFactory
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.common.deprecation import deprecated
def _disjunct_on_active_block(self, disjunct):
    parent_block = disjunct.parent_block()
    while parent_block is not None:
        if parent_block.ctype is Block and (not parent_block.active):
            return False
        elif parent_block.ctype is Disjunct and (not parent_block.active) and (parent_block.indicator_var.value == False) and parent_block.indicator_var.fixed:
            return False
        else:
            parent_block = parent_block.parent_block()
            continue
    return True