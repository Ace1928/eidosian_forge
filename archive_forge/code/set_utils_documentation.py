from collections import Counter
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block
from pyomo.core.base.set import SetProduct

    Finds any block or constraint in block b, indexed explicitly (and not
    implicitly) by cset, and deactivates it at points specified.
    Implicitly indexed components are excluded because one of their parent
    blocks will be deactivated, so deactivating them too would be redundant.

    Args:
        b : Block to search
        cset : ContinuousSet of interest
        pts : Value or list of values, in ContinuousSet, to deactivate at

    Returns:
        A dictionary mapping points in pts to lists of
        component data that have been deactivated there
    