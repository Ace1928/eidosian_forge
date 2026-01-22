from io import StringIO
from pyomo.common.gc_manager import PauseGC
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
import logging
class ToGamsVisitor(_ToStringVisitor):
    _expression_handlers = {EXPR.PowExpression: _handle_PowExpression, EXPR.UnaryFunctionExpression: _handle_UnaryFunctionExpression, EXPR.AbsExpression: _handle_AbsExpression}

    def __init__(self, smap, treechecker, output_fixed_variables=False):
        super(ToGamsVisitor, self).__init__(False, smap)
        self.treechecker = treechecker
        self.is_discontinuous = False
        self.output_fixed_variables = output_fixed_variables

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in native_types:
            try:
                return (True, ftoa(node, True))
            except TypeError:
                return (True, repr(node))
        if node.is_expression_type():
            if not node.is_potentially_variable():
                return (True, ftoa(node(), True))
            if node.__class__ is EXPR.MonomialTermExpression:
                return (True, self._monomial_to_string(node))
            if node.__class__ is EXPR.LinearExpression:
                return (True, self._linear_to_string(node))
            if node.is_component_type():
                self.treechecker(node)
            return (False, None)
        if node.is_component_type():
            if node.ctype not in valid_expr_ctypes_minlp:
                raise RuntimeError("Unallowable component '%s' of type %s found in an active constraint or objective.\nThe GAMS writer cannot export expressions with this component type." % (node.name, node.ctype.__name__))
            if node.ctype is not Var:
                self.treechecker(node)
        if node.is_fixed() and (not (self.output_fixed_variables and node.is_potentially_variable())):
            return (True, ftoa(node(), True))
        else:
            assert node.is_variable_type()
            return (True, self.smap.getSymbol(node))

    def _monomial_to_string(self, node):
        const, var = node.args
        if const.__class__ not in native_types:
            const = value(const)
        if var.is_fixed() and (not self.output_fixed_variables):
            return ftoa(const * var.value, True)
        if not const:
            return '0'
        if const in _plusMinusOne:
            if const < 0:
                return '-' + self.smap.getSymbol(var)
            else:
                return self.smap.getSymbol(var)
        return ftoa(const, True) + '*' + self.smap.getSymbol(var)

    def _linear_to_string(self, node):
        values = [self._monomial_to_string(arg) if arg.__class__ is EXPR.MonomialTermExpression else ftoa(arg, True) for arg in node.args]
        return node._to_string(values, False, self.smap)