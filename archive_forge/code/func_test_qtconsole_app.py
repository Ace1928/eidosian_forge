from pyomo.environ import (
import pyomo.common.unittest as unittest
import pyomo.contrib.viewer.qt as myqt
import pyomo.contrib.viewer.pyomo_viewer as pv
from pyomo.contrib.viewer.qt import available
@unittest.skipIf(not available, 'Qt packages are not available.')
def test_qtconsole_app(qtbot):
    app = pv.QtApp()
    app.initialize([])
    app.show_ui()
    app.hide_ui()