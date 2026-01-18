import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataItem
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.dependencies import DeferredImportError
def test_var_fixed_bounds(self):
    cdi = ComponentDataItem(parent=None, ui_data=None, o=self.m.x[1])
    cdi.set('fixed', True)
    self.assertTrue(cdi.get('fixed'))
    cdi.set('fixed', False)
    self.assertFalse(cdi.get('fixed'))