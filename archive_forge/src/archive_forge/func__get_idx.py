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
def _get_idx(l, ds, n, i, k):
    """
    This function returns the appropriate index for a variable
    indexed by a differential set. It's needed because the collocation
    constraints are indexed by finite element and collocation point
    however a ContinuousSet contains a list of all the discretization
    points and is not separated into finite elements and collocation
    points.
    """
    t = list(ds)
    tmp = ds.ord(ds._fe[i]) - 1
    tik = t[tmp + k]
    if n is None:
        return tik
    else:
        tmpn = n
        if not isinstance(n, tuple):
            tmpn = (n,)
    return tmpn[0:l] + (tik,) + tmpn[l:]