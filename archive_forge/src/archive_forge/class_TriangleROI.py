import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class TriangleROI(ROI):
    """
    Equilateral triangle ROI subclass with one scale handle and one rotation handle.
    Arguments
    pos            (length-2 sequence) The position of the ROI's origin.
    size           (float) The length of an edge of the triangle.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """

    def __init__(self, pos, size, **args):
        ROI.__init__(self, pos, [size, size], aspectLocked=True, **args)
        angles = np.linspace(0, np.pi * 4 / 3, 3)
        verticies = (np.array((np.sin(angles), np.cos(angles))).T + 1.0) / 2.0
        self.poly = QtGui.QPolygonF()
        for pt in verticies:
            self.poly.append(QtCore.QPointF(*pt))
        self.addRotateHandle(verticies[0], [0.5, 0.5])
        self.addScaleHandle(verticies[1], [0.5, 0.5])

    def paint(self, p, *args):
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.scale(r.width(), r.height())
        p.setPen(self.currentPen)
        p.drawPolygon(self.poly)

    def shape(self):
        self.path = QtGui.QPainterPath()
        r = self.boundingRect()
        t = QtGui.QTransform()
        t.scale(r.width(), r.height())
        self.path.addPolygon(self.poly)
        return t.map(self.path)

    def getArrayRegion(self, *args, **kwds):
        return self._getArrayRegionForArbitraryShape(*args, **kwds)