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
Convert CNF logical constraint to linear constraints.

    Expected expression node types: AndExpression, OrExpression, NotExpression,
    AtLeastExpression, AtMostExpression, ExactlyExpression, _BooleanVarData

    