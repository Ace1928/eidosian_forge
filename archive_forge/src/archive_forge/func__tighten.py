from pyomo.core import Var
from pyomo.core.base.indexed_component import UnindexedComponent_set
def _tighten(src, dest):
    starting_lb = dest.lb
    starting_ub = dest.ub
    if not src.is_continuous():
        dest.domain = src.domain
    if src.lb is not None:
        if starting_lb is None:
            dest.setlb(src.lb)
        else:
            dest.setlb(max(starting_lb, src.lb))
    if src.ub is not None:
        if starting_ub is None:
            dest.setub(src.ub)
        else:
            dest.setub(min(starting_ub, src.ub))