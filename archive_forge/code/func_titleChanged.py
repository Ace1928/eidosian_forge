from ...Qt import QtCore, QtWidgets, QtGui
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
def titleChanged(self):
    self.setSizeHint(0, self.button.sizeHint())