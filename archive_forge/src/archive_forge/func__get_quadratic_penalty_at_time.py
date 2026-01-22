from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def _get_quadratic_penalty_at_time(var, t, target, weight=None):
    if weight is None:
        weight = 1.0
    return weight * (var[t] - target) ** 2