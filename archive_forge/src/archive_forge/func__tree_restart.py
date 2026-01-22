import os
import logging
import pyomo.contrib.viewer.report as rpt
import pyomo.environ as pyo
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.model_browser import ModelBrowser
from pyomo.contrib.viewer.residual_table import ResidualTable
from pyomo.contrib.viewer.model_select import ModelSelect
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.fileutils import this_file_dir
def _tree_restart(self, w, cls=ModelBrowser, **kwargs):
    """
        Start/Restart a tree window
        """
    try:
        self._refresh_list.remove(w)
    except ValueError:
        pass
    try:
        try:
            self.mdiArea.removeSubWindow(w.parent())
        except RuntimeError:
            pass
        del w
        w = None
    except AttributeError:
        pass
    w = cls(**kwargs)
    self.mdiArea.addSubWindow(w)
    w.parent().show()
    self._refresh_list.append(w)
    return w