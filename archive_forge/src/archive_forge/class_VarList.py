import logging
import sys
from pyomo.common.pyomo_typing import overload
from weakref import ref as weakref_ref
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr import GetItemExpression
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.core.expr.numvalue import (
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.indexed_component import (
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.core.base.units_container import units
@ModelComponentFactory.register('List of decision variables.')
class VarList(IndexedVar):
    """
    Variable-length indexed variable objects used to construct Pyomo models.
    """

    def __init__(self, **kwargs):
        self._starting_index = kwargs.pop('starting_index', 1)
        args = (Set(dimen=1),)
        IndexedVar.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """Construct this component."""
        if self._constructed:
            return
        if is_debug_set(logger):
            logger.debug('Constructing variable list %s', self.name)
        self.index_set().construct()
        if self._rule_init is not None and self._rule_init.contains_indices():
            for i, idx in enumerate(self._rule_init.indices()):
                self._index_set.add(i + self._starting_index)
        super(VarList, self).construct(data)

    def add(self):
        """Add a variable to this list."""
        next_idx = len(self._index_set) + self._starting_index
        self._index_set.add(next_idx)
        return self[next_idx]