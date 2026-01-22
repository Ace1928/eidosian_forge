import builtins
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
def addChanged(self):
    """Called when "add new" combo is changed
        The parameter MUST have an 'addNew' method defined.
        """
    if self.addWidget.currentIndex() == 0:
        return
    typ = self.addWidget.currentText()
    self.param.addNew(typ)
    self.addWidget.setCurrentIndex(0)