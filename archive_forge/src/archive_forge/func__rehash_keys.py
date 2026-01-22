from collections.abc import MutableSet as collections_MutableSet
from collections.abc import Set as collections_Set
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections.component_map import _hasher
def _rehash_keys(encode, val):
    if encode:
        return val
    else:
        return {_hasher[obj.__class__](obj): obj for obj in val.values()}