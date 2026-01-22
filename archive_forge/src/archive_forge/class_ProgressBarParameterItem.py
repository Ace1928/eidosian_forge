from ...Qt import QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem
class ProgressBarParameterItem(WidgetParameterItem):

    def makeWidget(self):
        w = QtWidgets.QProgressBar()
        w.setMaximumHeight(20)
        w.sigChanged = w.valueChanged
        self.hideWidget = False
        return w