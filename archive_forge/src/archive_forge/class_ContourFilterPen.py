from __future__ import annotations
from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen
from fontTools.pens.recordingPen import RecordingPen
class ContourFilterPen(_PassThruComponentsMixin, RecordingPen):
    """A "buffered" filter pen that accumulates contour data, passes
    it through a ``filterContour`` method when the contour is closed or ended,
    and finally draws the result with the output pen.

    Components are passed through unchanged.
    """

    def __init__(self, outPen):
        super(ContourFilterPen, self).__init__()
        self._outPen = outPen

    def closePath(self):
        super(ContourFilterPen, self).closePath()
        self._flushContour()

    def endPath(self):
        super(ContourFilterPen, self).endPath()
        self._flushContour()

    def _flushContour(self):
        result = self.filterContour(self.value)
        if result is not None:
            self.value = result
        self.replay(self._outPen)
        self.value = []

    def filterContour(self, contour):
        """Subclasses must override this to perform the filtering.

        The contour is a list of pen (operator, operands) tuples.
        Operators are strings corresponding to the AbstractPen methods:
        "moveTo", "lineTo", "curveTo", "qCurveTo", "closePath" and
        "endPath". The operands are the positional arguments that are
        passed to each method.

        If the method doesn't return a value (i.e. returns None), it's
        assumed that the argument was modified in-place.
        Otherwise, the return value is drawn with the output pen.
        """
        return