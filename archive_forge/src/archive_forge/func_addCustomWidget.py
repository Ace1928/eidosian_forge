import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
def addCustomWidget(self, widgetClass, baseClass, module):
    assert widgetClass not in self._widgets
    self._widgets[widgetClass] = (baseClass, module)