import numpy as np
from ... import ComboBox, PlotDataItem
from ...graphicsItems.ScatterPlotItem import ScatterPlotItem
from ...Qt import QtCore, QtGui, QtWidgets
from ..Node import Node
from .common import CtrlNode
class PlotWidgetNode(Node):
    """Connection to PlotWidget. Will plot arrays, metaarrays, and display event lists."""
    nodeName = 'PlotWidget'
    sigPlotChanged = QtCore.Signal(object)

    def __init__(self, name):
        Node.__init__(self, name, terminals={'In': {'io': 'in', 'multi': True}})
        self.plot = None
        self.plots = {}
        self.ui = None
        self.items = {}

    def disconnected(self, localTerm, remoteTerm):
        if localTerm is self['In'] and remoteTerm in self.items:
            self.plot.removeItem(self.items[remoteTerm])
            del self.items[remoteTerm]

    def setPlot(self, plot):
        if plot == self.plot:
            return
        if self.plot is not None:
            for vid in list(self.items.keys()):
                self.plot.removeItem(self.items[vid])
                del self.items[vid]
        self.plot = plot
        self.updateUi()
        self.update()
        self.sigPlotChanged.emit(self)

    def getPlot(self):
        return self.plot

    def process(self, In, display=True):
        if display and self.plot is not None:
            items = set()
            for name, vals in In.items():
                if vals is None:
                    continue
                if type(vals) is not list:
                    vals = [vals]
                for val in vals:
                    vid = id(val)
                    if vid in self.items and self.items[vid].scene() is self.plot.scene():
                        items.add(vid)
                    else:
                        if isinstance(val, QtWidgets.QGraphicsItem):
                            self.plot.addItem(val)
                            item = val
                        else:
                            item = self.plot.plot(val)
                        self.items[vid] = item
                        items.add(vid)
            for vid in list(self.items.keys()):
                if vid not in items:
                    self.plot.removeItem(self.items[vid])
                    del self.items[vid]

    def processBypassed(self, args):
        if self.plot is None:
            return
        for item in list(self.items.values()):
            self.plot.removeItem(item)
        self.items = {}

    def ctrlWidget(self):
        if self.ui is None:
            self.ui = ComboBox()
            self.ui.currentIndexChanged.connect(self.plotSelected)
            self.updateUi()
        return self.ui

    def plotSelected(self, index):
        self.setPlot(self.ui.value())

    def setPlotList(self, plots):
        """
        Specify the set of plots (PlotWidget or PlotItem) that the user may
        select from.
        
        *plots* must be a dictionary of {name: plot} pairs.
        """
        self.plots = plots
        self.updateUi()

    def updateUi(self):
        self.ui.setItems(self.plots)
        try:
            self.ui.setValue(self.plot)
        except ValueError:
            pass