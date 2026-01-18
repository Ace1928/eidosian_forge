import math
import pickle
from pyomo.common.errors import PyomoException
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.util.check_units import assert_units_consistent, check_units_equivalent
from pyomo.core.expr import inequality
from pyomo.core.expr.numvalue import NumericConstant
import pyomo.core.expr as EXPR
from pyomo.core.base.units_container import (
from io import StringIO
def test_var_set_value(self):
    m = ConcreteModel()
    m.x = Var()
    m.x.value = 10
    self.assertEqual(m.x.value, 10)
    m.x.value = 20 * units.kg
    self.assertEqual(m.x.value, 20)
    m.x.value = 30 * units.dimensionless
    self.assertEqual(m.x.value, 30)
    del m.x
    m.x = Var(units=units.dimensionless)
    m.x.value = 10
    self.assertEqual(m.x.value, 10)
    with self.assertRaisesRegex(UnitsError, 'Cannot convert kg to dimensionless'):
        m.x.value = 20 * units.kg
    self.assertEqual(m.x.value, 10)
    m.x.value = 30 * units.dimensionless
    self.assertEqual(m.x.value, 30)
    del m.x
    m.x = Var(units=units.gram)
    m.x.value = 10
    self.assertEqual(m.x.value, 10)
    m.x.value = 20 * units.kg
    self.assertEqual(m.x.value, 20000)
    with self.assertRaisesRegex(UnitsError, 'Cannot convert dimensionless to g'):
        m.x.value = 30 * units.dimensionless
    self.assertEqual(m.x.value, 20000)
    del m.x