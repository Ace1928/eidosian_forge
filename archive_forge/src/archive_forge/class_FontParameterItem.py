from ...Qt import QtGui, QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem
class FontParameterItem(WidgetParameterItem):

    def makeWidget(self):
        w = QtWidgets.QFontComboBox()
        w.setMaximumHeight(20)
        w.sigChanged = w.currentFontChanged
        w.value = w.currentFont
        w.setValue = w.setCurrentFont
        self.hideWidget = False
        return w

    def updateDisplayLabel(self, value=None):
        if value is None:
            value = self.widget.currentText()
        super().updateDisplayLabel(value)