from collections.abc import MutableSet as collections_MutableSet
from collections.abc import Set as collections_Set
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections.component_map import _hasher
Remove an element. If not a member, raise a KeyError.