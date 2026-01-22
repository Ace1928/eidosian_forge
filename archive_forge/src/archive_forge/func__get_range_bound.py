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
def _get_range_bound(self, range_arg):
    if self._expr is None:
        return None
    bound = self._expr.arg(range_arg)
    if not is_fixed(bound):
        raise ValueError("Constraint '%s' is a Ranged Inequality with a variable %s bound.  Cannot normalize the constraint or send it to a solver." % (self.name, {0: 'lower', 2: 'upper'}[range_arg]))
    return bound