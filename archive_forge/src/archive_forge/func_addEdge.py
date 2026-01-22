import sys
import weakref
import math
from PySide2 import QtCore, QtGui, QtWidgets
def addEdge(self, edge):
    self.edgeList.append(weakref.ref(edge))
    edge.adjust()