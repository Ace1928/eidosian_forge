from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
@m.Disjunct()
def Y2(d):
    m = d.model()
    d.c = Constraint(expr=m.x == 9)