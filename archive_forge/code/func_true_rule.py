from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def true_rule(d, s):
    return m.a[s] == 0