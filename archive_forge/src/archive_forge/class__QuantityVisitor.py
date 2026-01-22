import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
class _QuantityVisitor(ExpressionValueVisitor):

    def __init__(self):
        self.native_types = set(nonpyomo_leaf_types)
        self.native_types.add(units._pint_registry.Quantity)
        self._unary_inverse_trig = {'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh'}

    def visit(self, node, values):
        """Visit nodes that have been expanded"""
        if node.__class__ in self.handlers:
            return self.handlers[node.__class__](self, node, values)
        return node._apply_operation(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in self.native_types:
            return (True, node)
        if node.is_expression_type():
            return (False, None)
        if node.is_numeric_type():
            if hasattr(node, 'get_units'):
                unit = node.get_units()
                if unit is not None:
                    return (True, value(node) * unit._pint_unit)
                else:
                    return (True, value(node))
            elif node.__class__ is _PyomoUnit:
                return (True, node._pint_unit)
            else:
                return (True, value(node))
        elif node.is_logical_type():
            return (True, value(node))
        else:
            return (True, node)

    def finalize(self, val):
        if val.__class__ is units._pint_registry.Quantity:
            return val
        elif val.__class__ is units._pint_registry.Unit:
            return 1.0 * val
        try:
            return val * units._pint_dimensionless
        except:
            return val

    def _handle_unary_function(self, node, values):
        ans = node._apply_operation(values)
        if node.getname() in self._unary_inverse_trig:
            ans = ans * units._pint_registry.radian
        return ans

    def _handle_external(self, node, values):
        ans = node._apply_operation([val.magnitude if val.__class__ is units._pint_registry.Quantity else val for val in values])
        unit = node.get_units()
        if unit is not None:
            ans = ans * unit._pint_unit
        return ans