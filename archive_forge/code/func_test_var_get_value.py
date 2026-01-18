import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataItem
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.dependencies import DeferredImportError
def test_var_get_value(self):
    cdi = ComponentDataItem(parent=None, ui_data=None, o=self.m.x[1])
    self.assertAlmostEqual(cdi.get('value'), 1)
    self.assertIsNone(cdi.get(expr))