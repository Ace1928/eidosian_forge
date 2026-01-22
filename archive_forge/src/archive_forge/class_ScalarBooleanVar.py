import logging
from weakref import ref as weakref_ref, ReferenceType
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.boolean_value import BooleanValue
from pyomo.core.expr import GetItemExpression
from pyomo.core.expr.numvalue import value
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Set, BooleanSet, Binary
from pyomo.core.base.util import is_functor
from pyomo.core.base.var import Var
class ScalarBooleanVar(_GeneralBooleanVarData, BooleanVar):
    """A single variable."""

    def __init__(self, *args, **kwd):
        _GeneralBooleanVarData.__init__(self, component=self)
        BooleanVar.__init__(self, *args, **kwd)
        self._index = UnindexedComponent_index

    @property
    def value(self):
        """Return the value for this variable."""
        if self._constructed:
            return _GeneralBooleanVarData.value.fget(self)
        raise ValueError("Accessing the value of variable '%s' before the Var has been constructed (there is currently no value to return)." % self.name)

    @value.setter
    def value(self, val):
        """Set the value for this variable."""
        if self._constructed:
            return _GeneralBooleanVarData.value.fset(self, val)
        raise ValueError("Setting the value of variable '%s' before the Var has been constructed (there is currently nothing to set." % self.name)

    @property
    def domain(self):
        return _GeneralBooleanVarData.domain.fget(self)

    def fix(self, value=NOTSET, skip_validation=False):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        if self._constructed:
            return _GeneralBooleanVarData.fix(self, value, skip_validation)
        raise ValueError("Fixing variable '%s' before the Var has been constructed (there is currently nothing to set)." % self.name)

    def unfix(self):
        """Sets the fixed indicator to False."""
        if self._constructed:
            return _GeneralBooleanVarData.unfix(self)
        raise ValueError("Freeing variable '%s' before the Var has been constructed (there is currently nothing to set)." % self.name)