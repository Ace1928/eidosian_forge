from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def _op_mode(m, t):
    m.disj2[t].c1 = Constraint(expr=m.y[t] == 10)
    return [m.disj1[t], m.disj2[t]]