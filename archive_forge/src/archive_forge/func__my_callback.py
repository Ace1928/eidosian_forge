import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.expr.taylor_series import taylor_series_expansion
def _my_callback(cb_m, cb_opt, cb_where):
    if cb_where == gurobipy.GRB.Callback.MIPSOL:
        cb_opt.cbGetSolution(vars=[m.x, m.y])
        if m.y.value < (m.x.value - 2) ** 2 - 1e-06:
            cb_opt.cbLazy(_add_cut(m.x.value))