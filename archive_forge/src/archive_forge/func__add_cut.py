import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.expr.taylor_series import taylor_series_expansion
def _add_cut(xval):
    m.x.value = xval
    return m.cons.add(m.y >= taylor_series_expansion((m.x - 2) ** 2))