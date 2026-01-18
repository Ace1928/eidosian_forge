import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
def updateGraph(self):
    pg.GraphItem.setData(self, **self.data)
    for i, item in enumerate(self.textItems):
        item.setPos(*self.data['pos'][i])