import inspect
import weakref
from .Qt import QtCore, QtWidgets
def widgetChanged(self, *args):
    w = self.sender()
    n = self.widgetList[w]
    v1 = self.cache[n]
    v2 = self.readWidget(w)
    if v1 != v2:
        self.sigChanged.emit(self.widgetList[w], v2)