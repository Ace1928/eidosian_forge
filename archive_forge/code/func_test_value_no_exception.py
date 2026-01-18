import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.environ import (
import pyomo.contrib.viewer.report as rpt
import pyomo.contrib.viewer.ui_data as uidata
def test_value_no_exception(self):
    self.m.x[2] = 0
    v = rpt.value_no_exception(self.m.b1.e2, div0='I like to divide by zero')
    assert v == 'I like to divide by zero'
    self.m.x[2] = None
    v = rpt.value_no_exception(self.m.b1.e2, div0=None)
    assert v is None
    self.m.x[2] = 0.0
    v = rpt.value_no_exception(self.m.b1.e5)
    assert v is None
    self.m.x[2] = 2.0
    v = rpt.value_no_exception(self.m.b1.e2, div0=None)
    self.assertAlmostEqual(v, 1)