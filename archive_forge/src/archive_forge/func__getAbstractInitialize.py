from inspect import isroutine
from pyomo.core import Var, Objective, Constraint, Set, Param
def _getAbstractInitialize(comp):
    """
    Returns the initialization rule. If initialize is a container; return None;
    that information will be collected during construction.
    """
    if isroutine(comp.initialize):
        return comp.initialize
    else:
        return None