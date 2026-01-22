import operator
from fontTools.cu2qu import curve_to_quadratic, curves_to_quadratic
from fontTools.pens.basePen import decomposeSuperBezierSegment
from fontTools.pens.filterPen import FilterPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.pointPen import BasePointToSegmentPen
from fontTools.pens.pointPen import ReverseContourPointPen
def _flushContour(self, segments):
    assert len(segments) >= 1
    closed = segments[0][0] != 'move'
    new_segments = []
    prev_points = segments[-1][1]
    prev_on_curve = prev_points[-1][0]
    for segment_type, points in segments:
        if segment_type == 'curve':
            for sub_points in self._split_super_bezier_segments(points):
                on_curve, smooth, name, kwargs = sub_points[-1]
                bcp1, bcp2 = (sub_points[0][0], sub_points[1][0])
                cubic = [prev_on_curve, bcp1, bcp2, on_curve]
                quad = curve_to_quadratic(cubic, self.max_err, self.all_quadratic)
                if self.stats is not None:
                    n = str(len(quad) - 2)
                    self.stats[n] = self.stats.get(n, 0) + 1
                new_points = [(pt, False, None, {}) for pt in quad[1:-1]]
                new_points.append((on_curve, smooth, name, kwargs))
                if self.all_quadratic or len(new_points) == 2:
                    new_segments.append(['qcurve', new_points])
                else:
                    new_segments.append(['curve', new_points])
                prev_on_curve = sub_points[-1][0]
        else:
            new_segments.append([segment_type, points])
            prev_on_curve = points[-1][0]
    if closed:
        new_segments = new_segments[-1:] + new_segments[:-1]
    self._drawPoints(new_segments)