import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
def test_uninitialized_value_error_message(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2])
    m.x[1].set_value(5)
    msg = 'No value for uninitialized NumericValue'
    with self.assertRaisesRegex(ValueError, msg):
        pyo.value(1 + m.x[1] * m.x[2])