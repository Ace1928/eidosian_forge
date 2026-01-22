import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
class VContainer(SplitContainer):

    def __init__(self, area):
        SplitContainer.__init__(self, area, QtCore.Qt.Orientation.Vertical)

    def type(self):
        return 'vertical'

    def updateStretch(self):
        x = 0
        y = 0
        sizes = []
        for i in range(self.count()):
            wx, wy = self.widget(i).stretch()
            y += wy
            x = max(x, wx)
            sizes.append(wy)
        self.setStretch(x, y)
        tot = float(sum(sizes))
        if tot == 0:
            scale = 1.0
        else:
            scale = self.height() / tot
        self.setSizes([int(s * scale) for s in sizes])