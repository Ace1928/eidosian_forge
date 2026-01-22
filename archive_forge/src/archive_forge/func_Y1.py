from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
@m.Disjunct()
def Y1(d):
    m = d.model()
    d.c = Constraint(expr=(1.15, m.x, 8))
    d.disjunction = Disjunction(expr=[m.Z1, m.Z2])