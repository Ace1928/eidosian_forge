import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import RenamedClass
from pyomo.common.modeling import NOTSET
from pyomo.common.formatting import tabular_writer
from pyomo.common.timing import ConstructionTimer
from pyomo.common.numeric_types import (
import pyomo.core.expr as EXPR
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.base.initializer import Initializer
class _ExpressionData(numeric_expr.NumericValue):
    """
    An object that defines a named expression.

    Public Class Attributes
        expr       The expression owned by this data.
    """
    __slots__ = ()
    EXPRESSION_SYSTEM = EXPR.ExpressionType.NUMERIC
    PRECEDENCE = 0
    ASSOCIATIVITY = EXPR.OperatorAssociativity.NON_ASSOCIATIVE

    def __call__(self, exception=True):
        """Compute the value of this expression."""
        arg, = self._args_
        if arg.__class__ in native_types:
            return arg
        return arg(exception=exception)

    def is_named_expression_type(self):
        """A boolean indicating whether this in a named expression."""
        return True

    def is_expression_type(self, expression_system=None):
        """A boolean indicating whether this in an expression."""
        return expression_system is None or expression_system == self.EXPRESSION_SYSTEM

    def arg(self, index):
        if index != 0:
            raise KeyError('Invalid index for expression argument: %d' % index)
        return self._args_[0]

    @property
    def args(self):
        return self._args_

    def nargs(self):
        return 1

    def _to_string(self, values, verbose, smap):
        if verbose:
            return '%s{%s}' % (str(self), values[0])
        if self._args_[0] is None:
            return '%s{None}' % str(self)
        return values[0]

    def clone(self):
        """Return a clone of this expression (no-op)."""
        return self

    def _apply_operation(self, result):
        return result[0]

    def polynomial_degree(self):
        """A tuple of subexpressions involved in this expressions operation."""
        if self._args_[0] is None:
            return None
        return self.expr.polynomial_degree()

    def _compute_polynomial_degree(self, result):
        return result[0]

    def _is_fixed(self, values):
        return values[0]

    @property
    def expr(self):
        arg, = self._args_
        if arg is None:
            return None
        return as_numeric(arg)

    @expr.setter
    def expr(self, value):
        self.set_value(value)

    def set_value(self, expr):
        """Set the expression on this expression."""
        raise NotImplementedError

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        raise NotImplementedError

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        raise NotImplementedError

    def is_potentially_variable(self):
        return True