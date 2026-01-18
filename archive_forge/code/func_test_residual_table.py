from pyomo.environ import (
import pyomo.common.unittest as unittest
import pyomo.contrib.viewer.qt as myqt
import pyomo.contrib.viewer.pyomo_viewer as pv
from pyomo.contrib.viewer.qt import available
@unittest.skipIf(not available, 'Qt packages are not available.')
def test_residual_table(qtbot):
    m = get_model()
    mw, m = get_mainwindow(model=m, testing=True)
    mw.residuals_restart()
    mw.ui_data.calculate_expressions()
    mw.residuals.calculate()
    mw.residuals_restart()
    mw.residuals.sort()
    dm = mw.residuals.tableView.model()
    assert dm.data(dm.index(0, 0)) == 'c4'
    assert dm.data(dm.index(0, 1)) == 'Divide_by_0'
    assert dm.data(dm.index(0, 2)) == 'Divide_by_0'
    assert dm.data(dm.index(0, 3)) == 0
    assert dm.data(dm.index(0, 4)) == 0
    assert dm.data(dm.index(0, 5)) == True
    m.c4.deactivate()
    mw.residuals.sort()
    assert dm.data(dm.index(0, 0)) == 'c5'