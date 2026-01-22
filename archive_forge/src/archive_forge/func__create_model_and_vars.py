import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.network import Port, Arc
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.base.units_container import pint_available, UnitsError
from pyomo.util.check_units import (
def _create_model_and_vars(self):
    u = units
    m = ConcreteModel()
    m.dx = Var(units=u.m, initialize=0.10188943773836046)
    m.dy = Var(units=u.m, initialize=0.0)
    m.vx = Var(units=u.m / u.s, initialize=0.7071067769802851)
    m.vy = Var(units=u.m / u.s, initialize=0.7071067769802851)
    m.t = Var(units=u.s, bounds=(1e-05, 10.0), initialize=0.0024015570927624456)
    m.theta = Var(bounds=(0, 0.49 * 3.14), initialize=0.7853981693583533, units=u.radians)
    m.a = Param(initialize=-32.2, units=u.ft / u.s ** 2)
    m.x_unitless = Var()
    return m