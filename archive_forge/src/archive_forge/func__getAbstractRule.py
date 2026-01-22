from inspect import isroutine
from pyomo.core import Var, Objective, Constraint, Set, Param
def _getAbstractRule(comp):
    """
    Returns the rule defining this component
    """
    return comp.rule