from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def twoDisj_twoCircles_easy():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 8))
    m.y = Var(bounds=(0, 10))
    m.upper_circle = Disjunct()
    m.upper_circle.cons = Constraint(expr=(m.x - 1) ** 2 + (m.y - 6) ** 2 <= 2)
    m.lower_circle = Disjunct()
    m.lower_circle.cons = Constraint(expr=(m.x - 4) ** 2 + (m.y - 2) ** 2 <= 2)
    m.disjunction = Disjunction(expr=[m.upper_circle, m.lower_circle])
    m.obj = Objective(expr=m.x + m.y, sense=maximize)
    return m