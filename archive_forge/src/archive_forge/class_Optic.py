import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
class Optic(pg.GraphicsObject, ParamObj):
    sigStateChanged = QtCore.Signal()

    def __init__(self, gitem, **params):
        ParamObj.__init__(self)
        pg.GraphicsObject.__init__(self)
        self.gitem = gitem
        self.surfaces = gitem.surfaces
        gitem.setParentItem(self)
        self.roi = pg.ROI([0, 0], [1, 1])
        self.roi.addRotateHandle([1, 1], [0.5, 0.5])
        self.roi.setParentItem(self)
        defaults = {'pos': Point(0, 0), 'angle': 0}
        defaults.update(params)
        self._ior_cache = {}
        self._connRoiChanged = self.roi.sigRegionChanged.connect(self.roiChanged)
        self.setParams(**defaults)

    def updateTransform(self):
        self.setPos(0, 0)
        tr = QtGui.QTransform()
        self.setTransform(tr.translate(Point(self['pos'])).rotate(self['angle']))

    def setParam(self, param, val):
        ParamObj.setParam(self, param, val)

    def paramStateChanged(self):
        """Some parameters of the optic have changed."""
        self.gitem.setPos(Point(self['pos']))
        self.gitem.resetTransform()
        self.gitem.setRotation(self['angle'])
        try:
            if isinstance(self._connRoiChanged, QtCore.QMetaObject.Connection):
                self.roi.sigRegionChanged.disconnect(self._connRoiChanged)
            else:
                self.roi.sigRegionChanged.disconnect(self.roiChanged)
            br = self.gitem.boundingRect()
            o = self.gitem.mapToParent(br.topLeft())
            self.roi.setAngle(self['angle'])
            self.roi.setPos(o)
            self.roi.setSize([br.width(), br.height()])
        finally:
            self._connRoiChanged = self.roi.sigRegionChanged.connect(self.roiChanged)
        self.sigStateChanged.emit()

    def roiChanged(self, *args):
        pos = self.roi.pos()
        self.gitem.resetTransform()
        self.gitem.setRotation(self.roi.angle())
        br = self.gitem.boundingRect()
        o1 = self.gitem.mapToParent(br.topLeft())
        self.setParams(angle=self.roi.angle(), pos=pos + (self.gitem.pos() - o1))

    def boundingRect(self):
        return QtCore.QRectF()

    def paint(self, p, *args):
        pass

    def ior(self, wavelength):
        return GLASSDB.ior(self['glass'], wavelength)