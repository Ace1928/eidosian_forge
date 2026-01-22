from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables
from pyomo.environ import Constraint, value
def active_constraint_set(blk):
    """
    Return a set of active constraints in a model.

    Args:
        blk: a Pyomo block in which to look for constraints.
    Returns:
        (ComponentSet): Active equality constraints
    """
    return ComponentSet(blk.component_data_objects(Constraint, active=True))