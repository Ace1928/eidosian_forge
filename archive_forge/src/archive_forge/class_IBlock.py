import logging
import math
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.base import _no_ctype, _convert_ctype
from pyomo.core.kernel.heterogeneous_container import IHeterogeneousContainer
from pyomo.core.kernel.container_utils import define_simple_containers
class IBlock(IHeterogeneousContainer):
    """A generalized container that can store objects of
    any category type as attributes.
    """
    __slots__ = ()
    _child_storage_delimiter_string = '.'
    _child_storage_entry_string = '%s'

    def child(self, key):
        """Get the child object associated with a given
        storage key for this container.

        Raises:
            KeyError: if the argument is not a storage key
                for any children of this container
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(str(key))