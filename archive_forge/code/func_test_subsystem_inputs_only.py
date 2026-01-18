import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
def test_subsystem_inputs_only(self):
    m = _make_simple_model()
    cons = [m.con2, m.con3]
    block = create_subsystem_block(cons)
    self.assertEqual(len(block.vars), 0)
    self.assertEqual(len(block.input_vars), 4)
    self.assertEqual(len(block.cons), 2)
    self.assertEqual(len([v for v in block.component_data_objects(pyo.Var) if not v.fixed]), 4)
    block.input_vars.fix()
    self.assertEqual(len([v for v in block.component_data_objects(pyo.Var) if not v.fixed]), 0)
    var_set = ComponentSet([m.v1, m.v2, m.v3, m.v4])
    self.assertIs(block.cons[0], m.con2)
    self.assertIs(block.cons[1], m.con3)
    self.assertIn(block.input_vars[0], var_set)
    self.assertIn(block.input_vars[1], var_set)
    self.assertIn(block.input_vars[2], var_set)
    self.assertIn(block.input_vars[3], var_set)
    self.assertIsNot(block.model(), m)
    for comp in block.component_objects((pyo.Var, pyo.Constraint)):
        self.assertTrue(comp.is_reference())
        for data in comp.values():
            self.assertIs(data.model(), m)