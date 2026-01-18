import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
@unittest.skipUnless(find_GSL(), 'Could not find the AMPL GSL library')
@unittest.skipUnless(pyo.SolverFactory('ipopt').available(), 'ipopt is not available')
def test_with_external_function(self):
    m = self._make_model_with_external_functions()
    subsystem = ([m.con2, m.con3], [m.v2, m.v3])
    m.v1.set_value(0.5)
    block = create_subsystem_block(*subsystem)
    ipopt = pyo.SolverFactory('ipopt')
    with TemporarySubsystemManager(to_fix=list(block.input_vars.values())):
        ipopt.solve(block)
    self.assertEqual(m.v1.value, 0.5)
    self.assertFalse(m.v1.fixed)
    self.assertAlmostEqual(m.v2.value, 1.04816, delta=1e-05)
    self.assertAlmostEqual(m.v3.value, 1.34356, delta=1e-05)
    m_full = self._solve_ef_model_with_ipopt()
    self.assertAlmostEqual(m.v1.value, m_full.v1.value)
    self.assertAlmostEqual(m.v2.value, m_full.v2.value)
    self.assertAlmostEqual(m.v3.value, m_full.v3.value)