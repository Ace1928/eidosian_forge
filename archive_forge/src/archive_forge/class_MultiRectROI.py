import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class MultiRectROI(QtWidgets.QGraphicsObject):
    """
    Chain of rectangular ROIs connected by handles.

    This is generally used to mark a curved path through
    an image similarly to PolyLineROI. It differs in that each segment
    of the chain is rectangular instead of linear and thus has width.
    
    ============== =============================================================
    **Arguments**
    points         (list of length-2 sequences) The list of points in the path.
    width          (float) The width of the ROIs orthogonal to the path.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """
    sigRegionChangeFinished = QtCore.Signal(object)
    sigRegionChangeStarted = QtCore.Signal(object)
    sigRegionChanged = QtCore.Signal(object)

    def __init__(self, points, width, pen=None, **args):
        QtWidgets.QGraphicsObject.__init__(self)
        self.pen = pen
        self.roiArgs = args
        self.lines = []
        if len(points) < 2:
            raise Exception('Must start with at least 2 points')
        self.addSegment(points[1], connectTo=points[0], scaleHandle=True)
        for p in points[2:]:
            self.addSegment(p)

    def paint(self, *args):
        pass

    def boundingRect(self):
        return QtCore.QRectF()

    def roiChangedEvent(self):
        w = self.lines[0].state['size'][1]
        for l in self.lines[1:]:
            w0 = l.state['size'][1]
            if w == w0:
                continue
            l.scale([1.0, w / w0], center=[0.5, 0.5])
        self.sigRegionChanged.emit(self)

    def roiChangeStartedEvent(self):
        self.sigRegionChangeStarted.emit(self)

    def roiChangeFinishedEvent(self):
        self.sigRegionChangeFinished.emit(self)

    def getHandlePositions(self):
        """Return the positions of all handles in local coordinates."""
        pos = [self.mapFromScene(self.lines[0].getHandles()[0].scenePos())]
        for l in self.lines:
            pos.append(self.mapFromScene(l.getHandles()[1].scenePos()))
        return pos

    def getArrayRegion(self, arr, img=None, axes=(0, 1), **kwds):
        """
        Return the result of :meth:`~pyqtgraph.ROI.getArrayRegion` for each rect
        in the chain concatenated into a single ndarray.

        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the
        arguments.

        Note: ``returnMappedCoords`` is not yet supported for this ROI type.
        """
        rgns = []
        for l in self.lines:
            rgn = l.getArrayRegion(arr, img, axes=axes, **kwds)
            if rgn is None:
                continue
            rgns.append(rgn)
        if img.axisOrder == 'row-major':
            axes = axes[::-1]
        ms = min([r.shape[axes[1]] for r in rgns])
        sl = [slice(None)] * rgns[0].ndim
        sl[axes[1]] = slice(0, ms)
        rgns = [r[tuple(sl)] for r in rgns]
        return np.concatenate(rgns, axis=axes[0])

    def addSegment(self, pos=(0, 0), scaleHandle=False, connectTo=None):
        """
        Add a new segment to the ROI connecting from the previous endpoint to *pos*.
        (pos is specified in the parent coordinate system of the MultiRectROI)
        """
        if connectTo is None:
            connectTo = self.lines[-1].getHandles()[1]
        newRoi = ROI((0, 0), [1, 5], parent=self, pen=self.pen, **self.roiArgs)
        self.lines.append(newRoi)
        if isinstance(connectTo, Handle):
            self.lines[-1].addScaleRotateHandle([0, 0.5], [1, 0.5], item=connectTo)
            newRoi.movePoint(connectTo, connectTo.scenePos(), coords='scene')
        else:
            h = self.lines[-1].addScaleRotateHandle([0, 0.5], [1, 0.5])
            newRoi.movePoint(h, connectTo, coords='scene')
        h = self.lines[-1].addScaleRotateHandle([1, 0.5], [0, 0.5])
        newRoi.movePoint(h, pos)
        if scaleHandle:
            newRoi.addScaleHandle([0.5, 1], [0.5, 0.5])
        newRoi.translatable = False
        newRoi.sigRegionChanged.connect(self.roiChangedEvent)
        newRoi.sigRegionChangeStarted.connect(self.roiChangeStartedEvent)
        newRoi.sigRegionChangeFinished.connect(self.roiChangeFinishedEvent)
        self.sigRegionChanged.emit(self)

    def removeSegment(self, index=-1):
        """Remove a segment from the ROI."""
        roi = self.lines[index]
        self.lines.pop(index)
        self.scene().removeItem(roi)
        roi.sigRegionChanged.disconnect(self.roiChangedEvent)
        roi.sigRegionChangeStarted.disconnect(self.roiChangeStartedEvent)
        roi.sigRegionChangeFinished.disconnect(self.roiChangeFinishedEvent)
        self.sigRegionChanged.emit(self)