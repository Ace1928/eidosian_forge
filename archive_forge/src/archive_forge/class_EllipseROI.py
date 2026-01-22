import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class EllipseROI(ROI):
    """
    Elliptical ROI subclass with one scale handle and one rotation handle.


    ============== =============================================================
    **Arguments**
    pos            (length-2 sequence) The position of the ROI's origin.
    size           (length-2 sequence) The size of the ROI's bounding rectangle.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """

    def __init__(self, pos, size, **args):
        self.path = None
        ROI.__init__(self, pos, size, **args)
        self.sigRegionChanged.connect(self._clearPath)
        self._addHandles()

    def _addHandles(self):
        self.addRotateHandle([1.0, 0.5], [0.5, 0.5])
        self.addScaleHandle([0.5 * 2.0 ** (-0.5) + 0.5, 0.5 * 2.0 ** (-0.5) + 0.5], [0.5, 0.5])

    def _clearPath(self):
        self.path = None

    def paint(self, p, opt, widget):
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self.currentPen)
        p.scale(r.width(), r.height())
        r = QtCore.QRectF(r.x() / r.width(), r.y() / r.height(), 1, 1)
        p.drawEllipse(r)

    def getArrayRegion(self, arr, img=None, axes=(0, 1), returnMappedCoords=False, **kwds):
        """
        Return the result of :meth:`~pyqtgraph.ROI.getArrayRegion` masked by the
        elliptical shape of the ROI. Regions outside the ellipse are set to 0.

        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the
        arguments.

        Note: ``returnMappedCoords`` is not yet supported for this ROI type.
        """
        if returnMappedCoords:
            arr, mappedCoords = ROI.getArrayRegion(self, arr, img, axes, returnMappedCoords, **kwds)
        else:
            arr = ROI.getArrayRegion(self, arr, img, axes, returnMappedCoords, **kwds)
        if arr is None or arr.shape[axes[0]] == 0 or arr.shape[axes[1]] == 0:
            if returnMappedCoords:
                return (arr, mappedCoords)
            else:
                return arr
        w = arr.shape[axes[0]]
        h = arr.shape[axes[1]]
        mask = np.fromfunction(lambda x, y: np.hypot((x + 0.5) / (w / 2.0) - 1, (y + 0.5) / (h / 2.0) - 1) < 1, (w, h))
        if axes[0] > axes[1]:
            mask = mask.T
        shape = [n if i in axes else 1 for i, n in enumerate(arr.shape)]
        mask = mask.reshape(shape)
        if returnMappedCoords:
            return (arr * mask, mappedCoords)
        else:
            return arr * mask

    def shape(self):
        if self.path is None:
            path = QtGui.QPainterPath()
            br = self.boundingRect()
            center = br.center()
            r1 = br.width() / 2.0
            r2 = br.height() / 2.0
            theta = np.linspace(0, 2 * np.pi, 24)
            x = center.x() + r1 * np.cos(theta)
            y = center.y() + r2 * np.sin(theta)
            path.moveTo(x[0], y[0])
            for i in range(1, len(x)):
                path.lineTo(x[i], y[i])
            self.path = path
        return self.path