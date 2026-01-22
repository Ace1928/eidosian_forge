import os
from pyomo.common.dependencies import attempt_import, UnavailableClass
from pyomo.scripting.pyomo_parser import add_subparser
import pyomo.contrib.viewer.qt as myqt
def active_widget_name(self):
    current_widget = self.window.tab_widget.currentWidget()
    current_widget_index = self.window.tab_widget.indexOf(current_widget)
    return self.window.tab_widget.tabText(current_widget_index)