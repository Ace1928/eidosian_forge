import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core import Suffix, Var, Constraint, Piecewise, Block
from pyomo.core import Expression, Param
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.block import IndexedBlock, SortComponents
from pyomo.dae import ContinuousSet, DAE_Error
from pyomo.common.formatting import tostr
from io import StringIO
def _update_var(v):
    """
    This method will construct any additional indices in a variable
    resulting from the discretization of a ContinuousSet.
    """
    new_indices = set(v.index_set()) - set(v._data.keys())
    for index in new_indices:
        v.add(index)