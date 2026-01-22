import inspect
import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.deprecation import RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.boolean_value import as_boolean, BooleanConstant
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Set
def _check_skip_add(self, index, expr):
    _expr_type = expr.__class__
    if expr is None:
        raise ValueError(_rule_returned_none_error % (_get_indexed_component_data_name(self, index),))
    if expr is True:
        raise ValueError("LogicalConstraint '%s' is always True." % (_get_indexed_component_data_name(self, index),))
    if expr is False:
        raise ValueError("LogicalConstraint '%s' is always False." % (_get_indexed_component_data_name(self, index),))
    if _expr_type is tuple and len(expr) == 1:
        if expr is LogicalConstraint.Skip:
            return None
        if expr is LogicalConstraint.Infeasible:
            raise ValueError("LogicalConstraint '%s' cannot be passed 'Infeasible' as a value." % (_get_indexed_component_data_name(self, index),))
    return expr