import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
@unittest.skipUnless(pyo.SolverFactory('ipopt').available(), 'Ipopt is not available')
def test_with_solve(self):
    m = _make_simple_model()
    ipopt = pyo.SolverFactory('ipopt')
    n_scenario = 2
    input_values = ComponentMap([(m.v3, [1.3, 2.3]), (m.v4, [1.4, 2.4])])
    _v1_val_1 = pyo.sqrt(3 * 1.4 + 1.3)
    _v1_val_2 = pyo.sqrt(3 * 2.4 + 2.3)
    _v2_val_1 = pyo.sqrt(2 * 1.4 - _v1_val_1)
    _v2_val_2 = pyo.sqrt(2 * 2.4 - _v1_val_2)
    output_values = ComponentMap([(m.v1, [_v1_val_1, _v1_val_2]), (m.v2, [_v2_val_1, _v2_val_2])])
    to_fix = [m.v3, m.v4]
    to_deactivate = [m.con1]
    to_reset = [m.v1, m.v2]
    m.v1.set_value(1.0)
    m.v2.set_value(1.0)
    with ParamSweeper(n_scenario, input_values, output_values, to_fix=to_fix, to_deactivate=to_deactivate, to_reset=to_reset) as sweeper:
        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        self.assertFalse(m.con1.active)
        self.assertTrue(m.con2.active)
        self.assertTrue(m.con3.active)
        for i, (inputs, outputs) in enumerate(sweeper):
            ipopt.solve(m)
            for var, val in inputs.items():
                self.assertEqual(var.value, val)
                self.assertEqual(var.value, input_values[var][i])
            for var, val in outputs.items():
                self.assertAlmostEqual(var.value, val, delta=1e-08)
                self.assertAlmostEqual(var.value, output_values[var][i], delta=1e-08)
    self.assertIs(m.v1.value, 1.0)
    self.assertIs(m.v2.value, 1.0)
    self.assertIs(m.v3.value, None)
    self.assertIs(m.v4.value, None)