from pyomo.environ import (
import pyomo.common.unittest as unittest
import pyomo.contrib.viewer.qt as myqt
import pyomo.contrib.viewer.pyomo_viewer as pv
from pyomo.contrib.viewer.qt import available
@unittest.skipIf(not available, 'Qt packages are not available.')
def test_show_model_select_no_models(qtbot):
    mw, m = get_mainwindow(model=None, testing=True)
    ms = mw.show_model_select()
    ms.update_models()
    ms.select_model()