import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.util import set_var_valid_value
from pyomo.environ import Var, Integers, ConcreteModel, Integers
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_OA_config
from pyomo.contrib.mindtpy.tests.MINLP5_simple import SimpleMINLP5
from pyomo.contrib.mindtpy.util import add_var_bound
def test_set_var_valid_value(self):
    m = ConcreteModel()
    m.x1 = Var(within=Integers, bounds=(-1, 4), initialize=0)
    set_var_valid_value(m.x1, var_val=5, integer_tolerance=1e-06, zero_tolerance=1e-06, ignore_integrality=False)
    self.assertEqual(m.x1.value, 4)
    set_var_valid_value(m.x1, var_val=-2, integer_tolerance=1e-06, zero_tolerance=1e-06, ignore_integrality=False)
    self.assertEqual(m.x1.value, -1)
    set_var_valid_value(m.x1, var_val=1.1, integer_tolerance=1e-06, zero_tolerance=1e-06, ignore_integrality=True)
    self.assertEqual(m.x1.value, 1.1)
    set_var_valid_value(m.x1, var_val=2.00000001, integer_tolerance=1e-06, zero_tolerance=1e-06, ignore_integrality=False)
    self.assertEqual(m.x1.value, 2)
    set_var_valid_value(m.x1, var_val=1e-07, integer_tolerance=1e-09, zero_tolerance=1e-06, ignore_integrality=False)
    self.assertEqual(m.x1.value, 0)