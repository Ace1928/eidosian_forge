import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
class CustomTickSliderItem(pg.TickSliderItem):

    def __init__(self, *args, **kwds):
        pg.TickSliderItem.__init__(self, *args, **kwds)
        self.all_ticks = {}
        self._range = [0, 1]

    def setTicks(self, ticks):
        for tick, pos in self.listTicks():
            self.removeTick(tick)
        for pos in ticks:
            tickItem = self.addTick(pos, movable=False, color='#333333')
            self.all_ticks[pos] = tickItem
        self.updateRange(None, self._range)

    def updateRange(self, vb, viewRange):
        origin = self.tickSize / 2.0
        length = self.length
        lengthIncludingPadding = length + self.tickSize + 2
        self._range = viewRange
        for pos in self.all_ticks:
            tickValueIncludingPadding = (pos - viewRange[0]) / (viewRange[1] - viewRange[0])
            tickValue = (tickValueIncludingPadding * lengthIncludingPadding - origin) / length
            visible = bool(tickValue >= 0 and tickValue <= 1)
            tick = self.all_ticks[pos]
            tick.setVisible(visible)
            if visible:
                self.setTickValue(tick, tickValue)