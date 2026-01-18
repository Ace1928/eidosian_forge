import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.network import Port, Arc
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.base.units_container import pint_available, UnitsError
from pyomo.util.check_units import (
def test_units_roundoff_error(self):
    m = ConcreteModel()
    m.var_1 = Var(initialize=400, units=units.J ** 0.4 * units.kg ** 0.2 * units.W ** 0.6 / units.K / units.m ** 2.2 / units.Pa ** 0.2 / units.s ** 0.8)
    m.var_1.fix()
    m.var_2 = Var(initialize=400, units=units.kg / units.s ** 3 / units.K)
    assert_units_equivalent(m.var_1, m.var_2)