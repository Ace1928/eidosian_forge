from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets, QtGui
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
def updateMin(self):
    bounds = self.itemRect()
    self.setMinimumWidth(bounds.width())
    self.setMinimumHeight(bounds.height())
    self._sizeHint = {QtCore.Qt.SizeHint.MinimumSize: (bounds.width(), bounds.height()), QtCore.Qt.SizeHint.PreferredSize: (bounds.width(), bounds.height()), QtCore.Qt.SizeHint.MaximumSize: (-1, -1), QtCore.Qt.SizeHint.MinimumDescent: (0, 0)}
    self.updateGeometry()