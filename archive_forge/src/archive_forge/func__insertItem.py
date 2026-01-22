import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
def _insertItem(self, item, index):
    if not isinstance(item, Dock):
        raise Exception('Tab containers may hold only docks, not other containers.')
    self.stack.insertWidget(index, item)
    self.hTabLayout.insertWidget(index, item.label)
    item.label.sigClicked.connect(self.tabClicked)
    self.tabClicked(item.label)