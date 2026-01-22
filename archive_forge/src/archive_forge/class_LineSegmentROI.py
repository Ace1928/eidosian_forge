import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class LineSegmentROI(ROI):
    """
    ROI subclass with two freely-moving handles defining a line.

    ============== =============================================================
    **Arguments**
    positions      (list of two length-2 sequences) The endpoints of the line 
                   segment. Note that, unlike the handle positions specified in 
                   other ROIs, these positions must be expressed in the normal
                   coordinate system of the ROI, rather than (0 to 1) relative
                   to the size of the ROI.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """

    def __init__(self, positions=(None, None), pos=None, handles=(None, None), **args):
        if pos is None:
            pos = [0, 0]
        ROI.__init__(self, pos, [1, 1], **args)
        if len(positions) > 2:
            raise Exception('LineSegmentROI must be defined by exactly 2 positions. For more points, use PolyLineROI.')
        for i, p in enumerate(positions):
            self.addFreeHandle(p, item=handles[i])

    @property
    def endpoints(self):
        return [h['item'] for h in self.handles]

    def listPoints(self):
        return [p['item'].pos() for p in self.handles]

    def getState(self):
        state = ROI.getState(self)
        state['points'] = [Point(h.pos()) for h in self.getHandles()]
        return state

    def saveState(self):
        state = ROI.saveState(self)
        state['points'] = [tuple(h.pos()) for h in self.getHandles()]
        return state

    def setState(self, state):
        ROI.setState(self, state)
        p1 = [state['points'][0][0] + state['pos'][0], state['points'][0][1] + state['pos'][1]]
        p2 = [state['points'][1][0] + state['pos'][0], state['points'][1][1] + state['pos'][1]]
        self.movePoint(self.getHandles()[0], p1, finish=False)
        self.movePoint(self.getHandles()[1], p2)

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self.currentPen)
        h1 = self.endpoints[0].pos()
        h2 = self.endpoints[1].pos()
        p.drawLine(h1, h2)

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        p = QtGui.QPainterPath()
        h1 = self.endpoints[0].pos()
        h2 = self.endpoints[1].pos()
        dh = h2 - h1
        if dh.length() == 0:
            return p
        pxv = self.pixelVectors(dh)[1]
        if pxv is None:
            return p
        pxv *= 4
        p.moveTo(h1 + pxv)
        p.lineTo(h2 + pxv)
        p.lineTo(h2 - pxv)
        p.lineTo(h1 - pxv)
        p.lineTo(h1 + pxv)
        return p

    def getArrayRegion(self, data, img, axes=(0, 1), order=1, returnMappedCoords=False, **kwds):
        """
        Use the position of this ROI relative to an imageItem to pull a slice 
        from an array.
        
        Since this pulls 1D data from a 2D coordinate system, the return value 
        will have ndim = data.ndim-1
        
        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the
        arguments.
        """
        imgPts = [self.mapToItem(img, h.pos()) for h in self.endpoints]
        d = Point(imgPts[1] - imgPts[0])
        o = Point(imgPts[0])
        rgn = fn.affineSlice(data, shape=(int(d.length()),), vectors=[Point(d.norm())], origin=o, axes=axes, order=order, returnCoords=returnMappedCoords, **kwds)
        return rgn