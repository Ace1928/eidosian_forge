import math
from typing import Any, Optional, Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.basePen import AbstractPen, MissingComponentError, PenError
from fontTools.misc.transform import DecomposedTransform, Identity
class GuessSmoothPointPen(AbstractPointPen):
    """
    Filtering PointPen that tries to determine whether an on-curve point
    should be "smooth", ie. that it's a "tangent" point or a "curve" point.
    """

    def __init__(self, outPen, error=0.05):
        self._outPen = outPen
        self._error = error
        self._points = None

    def _flushContour(self):
        if self._points is None:
            raise PenError('Path not begun')
        points = self._points
        nPoints = len(points)
        if not nPoints:
            return
        if points[0][1] == 'move':
            indices = range(1, nPoints - 1)
        elif nPoints > 1:
            indices = range(-1, nPoints - 1)
        else:
            indices = []
        for i in indices:
            pt, segmentType, _, name, kwargs = points[i]
            if segmentType is None:
                continue
            prev = i - 1
            next = i + 1
            if points[prev][1] is not None and points[next][1] is not None:
                continue
            pt = points[i][0]
            prevPt = points[prev][0]
            nextPt = points[next][0]
            if pt != prevPt and pt != nextPt:
                dx1, dy1 = (pt[0] - prevPt[0], pt[1] - prevPt[1])
                dx2, dy2 = (nextPt[0] - pt[0], nextPt[1] - pt[1])
                a1 = math.atan2(dy1, dx1)
                a2 = math.atan2(dy2, dx2)
                if abs(a1 - a2) < self._error:
                    points[i] = (pt, segmentType, True, name, kwargs)
        for pt, segmentType, smooth, name, kwargs in points:
            self._outPen.addPoint(pt, segmentType, smooth, name, **kwargs)

    def beginPath(self, identifier=None, **kwargs):
        if self._points is not None:
            raise PenError('Path already begun')
        self._points = []
        if identifier is not None:
            kwargs['identifier'] = identifier
        self._outPen.beginPath(**kwargs)

    def endPath(self):
        self._flushContour()
        self._outPen.endPath()
        self._points = None

    def addPoint(self, pt, segmentType=None, smooth=False, name=None, identifier=None, **kwargs):
        if self._points is None:
            raise PenError('Path not begun')
        if identifier is not None:
            kwargs['identifier'] = identifier
        self._points.append((pt, segmentType, False, name, kwargs))

    def addComponent(self, glyphName, transformation, identifier=None, **kwargs):
        if self._points is not None:
            raise PenError('Components must be added before or after contours')
        if identifier is not None:
            kwargs['identifier'] = identifier
        self._outPen.addComponent(glyphName, transformation, **kwargs)

    def addVarComponent(self, glyphName, transformation, location, identifier=None, **kwargs):
        if self._points is not None:
            raise PenError('VarComponents must be added before or after contours')
        if identifier is not None:
            kwargs['identifier'] = identifier
        self._outPen.addVarComponent(glyphName, transformation, location, **kwargs)