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
class _GeneralLogicalConstraintData(_LogicalConstraintData):
    """
    This class defines the data for a single general logical constraint.

    Constructor arguments:
        component       The LogicalStatement object that owns this data.
        expr            The Pyomo expression stored in this logical constraint.

    Public class attributes:
        active          A boolean that is true if this logical constraint is
                            active in the model.
        expr            The Pyomo expression for this logical constraint

    Private class attributes:
        _component      The logical constraint component.
        _active         A boolean that indicates whether this data is active
    """
    __slots__ = ('_expr',)

    def __init__(self, expr=None, component=None):
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self._active = True
        self._expr = None
        if expr is not None:
            self.set_value(expr)

    @property
    def body(self):
        """Access the body of a logical constraint expression."""
        return self._expr

    @property
    def expr(self):
        """Return the expression associated with this logical constraint."""
        return self.get_value()

    def set_value(self, expr):
        """Set the expression on this logical constraint."""
        if expr is None:
            self._expr = BooleanConstant(True)
            return
        expr_type = type(expr)
        if expr_type in native_types and expr_type not in native_logical_types:
            msg = "LogicalStatement '%s' does not have a proper value. Found '%s'.\nExpecting a logical expression or Boolean value. Examples:\n   (m.Y1 & m.Y2).implies(m.Y3)\n   atleast(1, m.Y1, m.Y2)"
            raise ValueError(msg)
        self._expr = as_boolean(expr)

    def get_value(self):
        """Get the expression on this logical constraint."""
        return self._expr