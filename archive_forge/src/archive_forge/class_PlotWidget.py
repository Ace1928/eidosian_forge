from ..graphicsItems.PlotItem import PlotItem
from ..Qt import QtCore, QtWidgets
from .GraphicsView import GraphicsView
class PlotWidget(GraphicsView):
    sigRangeChanged = QtCore.Signal(object, object)
    sigTransformChanged = QtCore.Signal(object)
    '\n    :class:`GraphicsView <pyqtgraph.GraphicsView>` widget with a single \n    :class:`PlotItem <pyqtgraph.PlotItem>` inside.\n    \n    The following methods are wrapped directly from PlotItem: \n    :func:`addItem <pyqtgraph.PlotItem.addItem>`, \n    :func:`removeItem <pyqtgraph.PlotItem.removeItem>`, \n    :func:`clear <pyqtgraph.PlotItem.clear>`, \n    :func:`setAxisItems <pyqtgraph.PlotItem.setAxisItems>`,\n    :func:`setXRange <pyqtgraph.ViewBox.setXRange>`,\n    :func:`setYRange <pyqtgraph.ViewBox.setYRange>`,\n    :func:`setRange <pyqtgraph.ViewBox.setRange>`,\n    :func:`autoRange <pyqtgraph.ViewBox.autoRange>`,\n    :func:`setXLink <pyqtgraph.ViewBox.setXLink>`,\n    :func:`setYLink <pyqtgraph.ViewBox.setYLink>`,\n    :func:`viewRect <pyqtgraph.ViewBox.viewRect>`,\n    :func:`setMouseEnabled <pyqtgraph.ViewBox.setMouseEnabled>`,\n    :func:`enableAutoRange <pyqtgraph.ViewBox.enableAutoRange>`,\n    :func:`disableAutoRange <pyqtgraph.ViewBox.disableAutoRange>`,\n    :func:`setAspectLocked <pyqtgraph.ViewBox.setAspectLocked>`,\n    :func:`setLimits <pyqtgraph.ViewBox.setLimits>`,\n    :func:`register <pyqtgraph.ViewBox.register>`,\n    :func:`unregister <pyqtgraph.ViewBox.unregister>`\n    \n    \n    For all \n    other methods, use :func:`getPlotItem <pyqtgraph.PlotWidget.getPlotItem>`.\n    '

    def __init__(self, parent=None, background='default', plotItem=None, **kargs):
        self.plotItem = None
        'When initializing PlotWidget, *parent* and *background* are passed to \n        :func:`GraphicsWidget.__init__() <pyqtgraph.GraphicsWidget.__init__>`\n        and all others are passed\n        to :func:`PlotItem.__init__() <pyqtgraph.PlotItem.__init__>`.'
        GraphicsView.__init__(self, parent, background=background)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.enableMouse(False)
        if plotItem is None:
            self.plotItem = PlotItem(**kargs)
        else:
            self.plotItem = plotItem
        self.setCentralItem(self.plotItem)
        for m in ['addItem', 'removeItem', 'autoRange', 'clear', 'setAxisItems', 'setXRange', 'setYRange', 'setRange', 'setAspectLocked', 'setMouseEnabled', 'setXLink', 'setYLink', 'enableAutoRange', 'disableAutoRange', 'setLimits', 'register', 'unregister', 'viewRect']:
            setattr(self, m, getattr(self.plotItem, m))
        self.plotItem.sigRangeChanged.connect(self.viewRangeChanged)

    def close(self):
        self.plotItem.close()
        self.plotItem = None
        self.setParent(None)
        super(PlotWidget, self).close()

    def __getattr__(self, attr):
        if hasattr(self.plotItem, attr):
            m = getattr(self.plotItem, attr)
            if hasattr(m, '__call__'):
                return m
        raise AttributeError(attr)

    def viewRangeChanged(self, view, range):
        self.sigRangeChanged.emit(self, range)

    def widgetGroupInterface(self):
        return (None, PlotWidget.saveState, PlotWidget.restoreState)

    def saveState(self):
        return self.plotItem.saveState()

    def restoreState(self, state):
        return self.plotItem.restoreState(state)

    def getPlotItem(self):
        """Return the PlotItem contained within."""
        return self.plotItem