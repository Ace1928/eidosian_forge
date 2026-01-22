from pyomo.core.expr.numeric_expr import LinearExpression
import pyomo.environ as pyo
from pyomo.core import Objective
def _models_have_same_sense(models):
    """Check if every model in the provided dict has the same objective sense.

    Input:
        models (dict) -- Keys are scenario names, values are Pyomo
            ConcreteModel objects.
    Returns:
        is_minimizing (bool) -- True if and only if minimizing. None if the
            check fails.
        check (bool) -- True only if all the models have the same sense (or
            no models were provided)
    Raises:
        ValueError -- If any of the models has either none or multiple
            active objectives.
    """
    if len(models) == 0:
        return (True, True)
    senses = [find_active_objective(scenario).is_minimizing() for scenario in models.values()]
    sense = senses[0]
    check = all((val == sense for val in senses))
    if check:
        return (sense == pyo.minimize, check)
    return (None, check)