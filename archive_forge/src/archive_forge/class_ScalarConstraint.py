import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.deprecation import RenamedClass
from pyomo.common.errors import DeveloperError
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.expr import (
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.set import Set
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
class ScalarConstraint(_GeneralConstraintData, Constraint):
    """
    ScalarConstraint is the implementation representing a single,
    non-indexed constraint.
    """

    def __init__(self, *args, **kwds):
        _GeneralConstraintData.__init__(self, component=self, expr=None)
        Constraint.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

    @property
    def body(self):
        """Access the body of a constraint expression."""
        if not self._data:
            raise ValueError("Accessing the body of ScalarConstraint '%s' before the Constraint has been assigned an expression. There is currently nothing to access." % self.name)
        return _GeneralConstraintData.body.fget(self)

    @property
    def lower(self):
        """Access the lower bound of a constraint expression."""
        if not self._data:
            raise ValueError("Accessing the lower bound of ScalarConstraint '%s' before the Constraint has been assigned an expression. There is currently nothing to access." % self.name)
        return _GeneralConstraintData.lower.fget(self)

    @property
    def upper(self):
        """Access the upper bound of a constraint expression."""
        if not self._data:
            raise ValueError("Accessing the upper bound of ScalarConstraint '%s' before the Constraint has been assigned an expression. There is currently nothing to access." % self.name)
        return _GeneralConstraintData.upper.fget(self)

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        if not self._data:
            raise ValueError("Accessing the equality flag of ScalarConstraint '%s' before the Constraint has been assigned an expression. There is currently nothing to access." % self.name)
        return _GeneralConstraintData.equality.fget(self)

    @property
    def strict_lower(self):
        """A boolean indicating whether this constraint has a strict lower bound."""
        if not self._data:
            raise ValueError("Accessing the strict_lower flag of ScalarConstraint '%s' before the Constraint has been assigned an expression. There is currently nothing to access." % self.name)
        return _GeneralConstraintData.strict_lower.fget(self)

    @property
    def strict_upper(self):
        """A boolean indicating whether this constraint has a strict upper bound."""
        if not self._data:
            raise ValueError("Accessing the strict_upper flag of ScalarConstraint '%s' before the Constraint has been assigned an expression. There is currently nothing to access." % self.name)
        return _GeneralConstraintData.strict_upper.fget(self)

    def clear(self):
        self._data = {}

    def set_value(self, expr):
        """Set the expression on this constraint."""
        if not self._data:
            self._data[None] = self
        return super(ScalarConstraint, self).set_value(expr)

    def add(self, index, expr):
        """Add a constraint with a given index."""
        if index is not None:
            raise ValueError("ScalarConstraint object '%s' does not accept index values other than None. Invalid value: %s" % (self.name, index))
        self.set_value(expr)
        return self