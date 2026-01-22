import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import (
class ToolbarQt(ToolContainerBase, QtWidgets.QToolBar):

    def __init__(self, toolmanager, parent=None):
        ToolContainerBase.__init__(self, toolmanager)
        QtWidgets.QToolBar.__init__(self, parent)
        self.setAllowedAreas(QtCore.Qt.ToolBarArea(_to_int(QtCore.Qt.ToolBarArea.TopToolBarArea) | _to_int(QtCore.Qt.ToolBarArea.BottomToolBarArea)))
        message_label = QtWidgets.QLabel('')
        message_label.setAlignment(QtCore.Qt.AlignmentFlag(_to_int(QtCore.Qt.AlignmentFlag.AlignRight) | _to_int(QtCore.Qt.AlignmentFlag.AlignVCenter)))
        message_label.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored))
        self._message_action = self.addWidget(message_label)
        self._toolitems = {}
        self._groups = {}

    def add_toolitem(self, name, group, position, image_file, description, toggle):
        button = QtWidgets.QToolButton(self)
        if image_file:
            button.setIcon(NavigationToolbar2QT._icon(self, image_file))
        button.setText(name)
        if description:
            button.setToolTip(description)

        def handler():
            self.trigger_tool(name)
        if toggle:
            button.setCheckable(True)
            button.toggled.connect(handler)
        else:
            button.clicked.connect(handler)
        self._toolitems.setdefault(name, [])
        self._add_to_group(group, name, button, position)
        self._toolitems[name].append((button, handler))

    def _add_to_group(self, group, name, button, position):
        gr = self._groups.get(group, [])
        if not gr:
            sep = self.insertSeparator(self._message_action)
            gr.append(sep)
        before = gr[position]
        widget = self.insertWidget(before, button)
        gr.insert(position, widget)
        self._groups[group] = gr

    def toggle_toolitem(self, name, toggled):
        if name not in self._toolitems:
            return
        for button, handler in self._toolitems[name]:
            button.toggled.disconnect(handler)
            button.setChecked(toggled)
            button.toggled.connect(handler)

    def remove_toolitem(self, name):
        for button, handler in self._toolitems[name]:
            button.setParent(None)
        del self._toolitems[name]

    def set_message(self, s):
        self.widgetForAction(self._message_action).setText(s)