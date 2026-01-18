from pyomo.common.collections import ComponentMap
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.core.base.boolean_var import _DeprecatedImplicitAssociatedBinaryVariable
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import native_logical_types, value
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.util import target_list
def update_boolean_vars_from_binary(model, integer_tolerance=1e-05):
    """Updates all Boolean variables based on the value of their linked binary
    variables."""
    for boolean_var in model.component_data_objects(BooleanVar, descend_into=Block):
        binary_var = boolean_var.get_associated_binary()
        if binary_var is not None and binary_var.value is not None:
            if abs(binary_var.value - 1) <= integer_tolerance:
                boolean_var.value = True
            elif abs(binary_var.value) <= integer_tolerance:
                boolean_var.value = False
            else:
                raise ValueError('Binary variable has non-{0,1} value: %s = %s' % (binary_var.name, binary_var.value))
            boolean_var.stale = binary_var.stale