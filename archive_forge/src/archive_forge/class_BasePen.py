from typing import Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
class BasePen(DecomposingPen):
    """Base class for drawing pens. You must override _moveTo, _lineTo and
    _curveToOne. You may additionally override _closePath, _endPath,
    addComponent, addVarComponent, and/or _qCurveToOne. You should not
    override any other methods.
    """

    def __init__(self, glyphSet=None):
        super(BasePen, self).__init__(glyphSet)
        self.__currentPoint = None

    def _moveTo(self, pt):
        raise NotImplementedError

    def _lineTo(self, pt):
        raise NotImplementedError

    def _curveToOne(self, pt1, pt2, pt3):
        raise NotImplementedError

    def _closePath(self):
        pass

    def _endPath(self):
        pass

    def _qCurveToOne(self, pt1, pt2):
        """This method implements the basic quadratic curve type. The
        default implementation delegates the work to the cubic curve
        function. Optionally override with a native implementation.
        """
        pt0x, pt0y = self.__currentPoint
        pt1x, pt1y = pt1
        pt2x, pt2y = pt2
        mid1x = pt0x + 0.6666666666666666 * (pt1x - pt0x)
        mid1y = pt0y + 0.6666666666666666 * (pt1y - pt0y)
        mid2x = pt2x + 0.6666666666666666 * (pt1x - pt2x)
        mid2y = pt2y + 0.6666666666666666 * (pt1y - pt2y)
        self._curveToOne((mid1x, mid1y), (mid2x, mid2y), pt2)

    def _getCurrentPoint(self):
        """Return the current point. This is not part of the public
        interface, yet is useful for subclasses.
        """
        return self.__currentPoint

    def closePath(self):
        self._closePath()
        self.__currentPoint = None

    def endPath(self):
        self._endPath()
        self.__currentPoint = None

    def moveTo(self, pt):
        self._moveTo(pt)
        self.__currentPoint = pt

    def lineTo(self, pt):
        self._lineTo(pt)
        self.__currentPoint = pt

    def curveTo(self, *points):
        n = len(points) - 1
        assert n >= 0
        if n == 2:
            self._curveToOne(*points)
            self.__currentPoint = points[-1]
        elif n > 2:
            _curveToOne = self._curveToOne
            for pt1, pt2, pt3 in decomposeSuperBezierSegment(points):
                _curveToOne(pt1, pt2, pt3)
                self.__currentPoint = pt3
        elif n == 1:
            self.qCurveTo(*points)
        elif n == 0:
            self.lineTo(points[0])
        else:
            raise AssertionError("can't get there from here")

    def qCurveTo(self, *points):
        n = len(points) - 1
        assert n >= 0
        if points[-1] is None:
            x, y = points[-2]
            nx, ny = points[0]
            impliedStartPoint = (0.5 * (x + nx), 0.5 * (y + ny))
            self.__currentPoint = impliedStartPoint
            self._moveTo(impliedStartPoint)
            points = points[:-1] + (impliedStartPoint,)
        if n > 0:
            _qCurveToOne = self._qCurveToOne
            for pt1, pt2 in decomposeQuadraticSegment(points):
                _qCurveToOne(pt1, pt2)
                self.__currentPoint = pt2
        else:
            self.lineTo(points[0])