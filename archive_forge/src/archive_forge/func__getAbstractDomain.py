from inspect import isroutine
from pyomo.core import Var, Objective, Constraint, Set, Param
def _getAbstractDomain(comp):
    """
    Returns the domain of this component
    """
    return getattr(comp, 'domain', None)