import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
@unittest.skipUnless(pyo.SolverFactory('ipopt').available(), 'Ipopt is not available')
def test_solve_subsystem(self):
    m = _make_simple_model()
    ipopt = pyo.SolverFactory('ipopt')
    m.v5 = pyo.Var(initialize=1.0)
    m.c4 = pyo.Constraint(expr=m.v5 == 5.0)
    cons = [m.con2, m.con3]
    vars = [m.v1, m.v2]
    block = create_subsystem_block(cons, vars)
    m.v3.fix(1.0)
    m.v4.fix(2.0)
    m.v1.set_value(1.0)
    m.v2.set_value(1.0)
    ipopt.solve(block)
    self.assertAlmostEqual(m.v1.value, pyo.sqrt(7.0), delta=1e-08)
    self.assertAlmostEqual(m.v2.value, pyo.sqrt(4.0 - pyo.sqrt(7.0)), delta=1e-08)
    self.assertEqual(m.v5.value, 1.0)