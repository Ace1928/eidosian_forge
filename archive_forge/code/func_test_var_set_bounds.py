import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataItem
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.dependencies import DeferredImportError
def test_var_set_bounds(self):
    cdi = ComponentDataItem(parent=None, ui_data=None, o=self.m.x[1])
    cdi.set('lb', 2)
    cdi.set('ub', 8)
    self.assertAlmostEqual(cdi.get('lb'), 2)
    self.assertAlmostEqual(cdi.get('ub'), 8)
    cdi = ComponentDataItem(parent=None, ui_data=None, o=self.m.x)
    cdi.set('lb', 0)
    cdi.set('ub', 10)
    cdi = ComponentDataItem(parent=None, ui_data=None, o=self.m.x[1])
    self.assertAlmostEqual(cdi.get('lb'), 0)
    self.assertAlmostEqual(cdi.get('ub'), 10)