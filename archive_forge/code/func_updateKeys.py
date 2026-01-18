import numpy as np
from ... import ComboBox, PlotDataItem
from ...graphicsItems.ScatterPlotItem import ScatterPlotItem
from ...Qt import QtCore, QtGui, QtWidgets
from ..Node import Node
from .common import CtrlNode
def updateKeys(self, data):
    if isinstance(data, dict):
        keys = list(data.keys())
    elif isinstance(data, list) or isinstance(data, tuple):
        keys = data
    elif isinstance(data, np.ndarray) or isinstance(data, np.void):
        keys = data.dtype.names
    else:
        print('Unknown data type:', type(data), data)
        return
    for c in self.ctrls.values():
        c.blockSignals(True)
    for c in [self.ctrls['x'], self.ctrls['y'], self.ctrls['size']]:
        cur = str(c.currentText())
        c.clear()
        for k in keys:
            c.addItem(k)
            if k == cur:
                c.setCurrentIndex(c.count() - 1)
    for c in [self.ctrls['color'], self.ctrls['border']]:
        c.setArgList(keys)
    for c in self.ctrls.values():
        c.blockSignals(False)
    self.keys = keys