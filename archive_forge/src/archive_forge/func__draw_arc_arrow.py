from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def _draw_arc_arrow(self, inner_radius, outer_radius, startangle, endangle, color, border=None, shaft_height_ratio=0.4, head_length_ratio=0.5, orientation='right', colour=None, **kwargs):
    """Draw an arrow along an arc (PRIVATE)."""
    if colour is not None:
        color = colour
    strokecolor, color = _stroke_and_fill_colors(color, border)
    startangle, endangle = (min(startangle, endangle), max(startangle, endangle))
    if orientation != 'left' and orientation != 'right':
        raise ValueError(f"Invalid orientation {orientation!r}, should be 'left' or 'right'")
    angle = endangle - startangle
    middle_radius = 0.5 * (inner_radius + outer_radius)
    boxheight = outer_radius - inner_radius
    shaft_height = boxheight * shaft_height_ratio
    shaft_inner_radius = middle_radius - 0.5 * shaft_height
    shaft_outer_radius = middle_radius + 0.5 * shaft_height
    headangle_delta = max(0.0, min(abs(boxheight) * head_length_ratio / middle_radius, abs(angle)))
    if angle < 0:
        headangle_delta *= -1
    if orientation == 'right':
        headangle = endangle - headangle_delta
    else:
        headangle = startangle + headangle_delta
    if startangle <= endangle:
        headangle = max(min(headangle, endangle), startangle)
    else:
        headangle = max(min(headangle, startangle), endangle)
    if not (startangle <= headangle <= endangle or endangle <= headangle <= startangle):
        raise RuntimeError('Problem drawing arrow, invalid positions. Start angle: %s, Head angle: %s, End angle: %s, Angle: %s' % (startangle, headangle, endangle, angle))
    startcos, startsin = (cos(startangle), sin(startangle))
    headcos, headsin = (cos(headangle), sin(headangle))
    endcos, endsin = (cos(endangle), sin(endangle))
    x0, y0 = (self.xcenter, self.ycenter)
    if 0.5 >= abs(angle) and abs(headangle_delta) >= abs(angle):
        if orientation == 'right':
            x1, y1 = (x0 + inner_radius * startsin, y0 + inner_radius * startcos)
            x2, y2 = (x0 + outer_radius * startsin, y0 + outer_radius * startcos)
            x3, y3 = (x0 + middle_radius * endsin, y0 + middle_radius * endcos)
        else:
            x1, y1 = (x0 + inner_radius * endsin, y0 + inner_radius * endcos)
            x2, y2 = (x0 + outer_radius * endsin, y0 + outer_radius * endcos)
            x3, y3 = (x0 + middle_radius * startsin, y0 + middle_radius * startcos)
        return Polygon([x1, y1, x2, y2, x3, y3], strokeColor=border or color, fillColor=color, strokeLineJoin=1, strokewidth=0)
    elif orientation == 'right':
        p = ArcPath(strokeColor=strokecolor, fillColor=color, strokeLineJoin=1, strokewidth=0, **kwargs)
        p.addArc(self.xcenter, self.ycenter, shaft_inner_radius, 90 - headangle * 180 / pi, 90 - startangle * 180 / pi, moveTo=True)
        p.addArc(self.xcenter, self.ycenter, shaft_outer_radius, 90 - headangle * 180 / pi, 90 - startangle * 180 / pi, reverse=True)
        if abs(angle) < 0.5:
            p.lineTo(x0 + outer_radius * headsin, y0 + outer_radius * headcos)
            p.lineTo(x0 + middle_radius * endsin, y0 + middle_radius * endcos)
            p.lineTo(x0 + inner_radius * headsin, y0 + inner_radius * headcos)
        else:
            self._draw_arc_line(p, outer_radius, middle_radius, 90 - headangle * 180 / pi, 90 - endangle * 180 / pi)
            self._draw_arc_line(p, middle_radius, inner_radius, 90 - endangle * 180 / pi, 90 - headangle * 180 / pi)
        p.closePath()
        return p
    else:
        p = ArcPath(strokeColor=strokecolor, fillColor=color, strokeLineJoin=1, strokewidth=0, **kwargs)
        p.addArc(self.xcenter, self.ycenter, shaft_inner_radius, 90 - endangle * 180 / pi, 90 - headangle * 180 / pi, moveTo=True, reverse=True)
        p.addArc(self.xcenter, self.ycenter, shaft_outer_radius, 90 - endangle * 180 / pi, 90 - headangle * 180 / pi, reverse=False)
        if abs(angle) < 0.5:
            p.lineTo(x0 + outer_radius * headsin, y0 + outer_radius * headcos)
            p.lineTo(x0 + middle_radius * startsin, y0 + middle_radius * startcos)
            p.lineTo(x0 + inner_radius * headsin, y0 + inner_radius * headcos)
        else:
            self._draw_arc_line(p, outer_radius, middle_radius, 90 - headangle * 180 / pi, 90 - startangle * 180 / pi)
            self._draw_arc_line(p, middle_radius, inner_radius, 90 - startangle * 180 / pi, 90 - headangle * 180 / pi)
        p.closePath()
        return p