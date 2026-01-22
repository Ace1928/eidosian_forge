import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
class SplitContainer(Container, QtWidgets.QSplitter):
    """Horizontal or vertical splitter with some changes:
     - save/restore works correctly
    """
    sigStretchChanged = QtCore.Signal()

    def __init__(self, area, orientation):
        QtWidgets.QSplitter.__init__(self)
        self.setOrientation(orientation)
        Container.__init__(self, area)

    def _insertItem(self, item, index):
        self.insertWidget(index, item)
        item.show()

    def saveState(self):
        sizes = self.sizes()
        if all((x == 0 for x in sizes)):
            sizes = [10] * len(sizes)
        return {'sizes': sizes}

    def restoreState(self, state):
        sizes = state['sizes']
        self.setSizes(sizes)
        for i in range(len(sizes)):
            self.setStretchFactor(i, sizes[i])

    def childEvent(self, ev):
        super().childEvent(ev)
        Container.childEvent_(self, ev)