import copy
import itertools
from pyomo.common import DeveloperError
from pyomo.common.collections import Sequence
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_index
def wildcard_values(self, sort=SortComponents.UNSORTED):
    """Return an iterator over this slice"""
    return _IndexedComponent_slice_iter(self, sort=sort)