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
class CnfToLinearVisitor(StreamBasedExpressionVisitor):
    """Convert CNF logical constraint to linear constraints.

    Expected expression node types: AndExpression, OrExpression, NotExpression,
    AtLeastExpression, AtMostExpression, ExactlyExpression, _BooleanVarData

    """

    def __init__(self, indicator_var, binary_varlist):
        super(CnfToLinearVisitor, self).__init__()
        self._indicator = indicator_var
        self._binary_varlist = binary_varlist

    def exitNode(self, node, values):
        if type(node) == AndExpression:
            return list((v if type(v) in _numeric_relational_types else v == 1 for v in values))
        elif type(node) == OrExpression:
            return sum(values) >= 1
        elif type(node) == NotExpression:
            return 1 - values[0]
        sum_values = sum(values[1:])
        num_args = node.nargs() - 1
        if self._indicator is None:
            if type(node) == AtLeastExpression:
                return sum_values >= values[0]
            elif type(node) == AtMostExpression:
                return sum_values <= values[0]
            elif type(node) == ExactlyExpression:
                return sum_values == values[0]
        else:
            rhs_lb, rhs_ub = compute_bounds_on_expr(values[0])
            if rhs_lb == float('-inf') or rhs_ub == float('inf'):
                raise ValueError('Cannot generate linear constraints for %s([N, *logical_args]) with unbounded N. Detected %s <= N <= %s.' % (type(node).__name__, rhs_lb, rhs_ub))
            indicator_binary = self._indicator.get_associated_binary()
            if type(node) == AtLeastExpression:
                return [sum_values >= values[0] - rhs_ub * (1 - indicator_binary), sum_values <= values[0] - 1 + (-(rhs_lb - 1) + num_args) * indicator_binary]
            elif type(node) == AtMostExpression:
                return [sum_values <= values[0] + (-rhs_lb + num_args) * (1 - indicator_binary), sum_values >= values[0] + 1 - (rhs_ub + 1) * indicator_binary]
            elif type(node) == ExactlyExpression:
                less_than_binary = self._binary_varlist.add()
                more_than_binary = self._binary_varlist.add()
                return [sum_values <= values[0] + (-rhs_lb + num_args) * (1 - indicator_binary), sum_values >= values[0] - rhs_ub * (1 - indicator_binary), indicator_binary + less_than_binary + more_than_binary >= 1, sum_values <= values[0] - 1 + (-(rhs_lb - 1) + num_args) * (1 - less_than_binary), sum_values >= values[0] + 1 - (rhs_ub + 1) * (1 - more_than_binary)]
        if type(node) in _numeric_relational_types:
            raise MouseTrap(f"core.logical_to_linear does not support transforming LogicalConstraints with embedded relational expressions.  Found '{node}'.")
        else:
            raise DeveloperError(f'Unsupported node type {type(node)} encountered when transforming a CNF expression to its linear equivalent ({node}).')

    def beforeChild(self, node, child, child_idx):
        if type(node) in special_boolean_atom_types and child is node.args[0]:
            return (False, child)
        if type(child) in native_logical_types:
            return (False, int(child))
        if type(child) in native_types:
            return (False, child)
        if child.is_expression_type():
            return (True, None)
        if hasattr(child, 'get_associated_binary'):
            return (False, child.get_associated_binary())
        else:
            return (False, child)

    def finalizeResult(self, result):
        if type(result) is list:
            return result
        elif type(result) in _numeric_relational_types:
            return [result]
        else:
            return [result == 1]