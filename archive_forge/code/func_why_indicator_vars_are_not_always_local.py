from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def why_indicator_vars_are_not_always_local():
    m = ConcreteModel()
    m.x = Var(bounds=(1, 10))

    @m.Disjunct()
    def Z1(d):
        m = d.model()
        d.c = Constraint(expr=m.x >= 1.1)

    @m.Disjunct()
    def Z2(d):
        m = d.model()
        d.c = Constraint(expr=m.x >= 1.2)

    @m.Disjunct()
    def Y1(d):
        m = d.model()
        d.c = Constraint(expr=(1.15, m.x, 8))
        d.disjunction = Disjunction(expr=[m.Z1, m.Z2])

    @m.Disjunct()
    def Y2(d):
        m = d.model()
        d.c = Constraint(expr=m.x == 9)
    m.disjunction = Disjunction(expr=[m.Y1, m.Y2])
    m.logical_cons = LogicalConstraint(expr=m.Y2.indicator_var.implies(m.Z1.indicator_var.land(m.Z2.indicator_var)))
    m.obj = Objective(expr=m.x, sense=maximize)
    return m