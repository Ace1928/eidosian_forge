import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
class HContainer(SplitContainer):

    def __init__(self, area):
        SplitContainer.__init__(self, area, QtCore.Qt.Orientation.Horizontal)

    def type(self):
        return 'horizontal'

    def updateStretch(self):
        x = 0
        y = 0
        sizes = []
        for i in range(self.count()):
            wx, wy = self.widget(i).stretch()
            x += wx
            y = max(y, wy)
            sizes.append(wx)
        self.setStretch(x, y)
        tot = float(sum(sizes))
        if tot == 0:
            scale = 1.0
        else:
            scale = self.width() / tot
        self.setSizes([int(s * scale) for s in sizes])