import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataItem
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.dependencies import DeferredImportError
def test_set_func(self):
    cdi = ComponentDataItem(parent=None, ui_data=None, o=self.m.x[1])
    self.assertIsNone(cdi.set('test_val', 5))
    self.assertIsNone(cdi.get('test_val'))
    cdi = ComponentDataItem(parent=None, ui_data=None, o=self.m.x)
    self.assertIsNone(cdi.set('test_val', 5))
    self.assertEqual(cdi.get('test_val'), 5)