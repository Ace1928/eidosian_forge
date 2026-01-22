import copy
import itertools
from pyomo.common import DeveloperError
from pyomo.common.collections import Sequence
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_index
def _advance_iter(_iter):
    return next(_iter)