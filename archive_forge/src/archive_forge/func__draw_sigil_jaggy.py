from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def _draw_sigil_jaggy(self, bottom, center, top, startangle, endangle, strand, color, border=None, **kwargs):
    """Draw JAGGY sigil (PRIVATE).

        Although we may in future expose the head/tail jaggy lengths, for now
        both the left and right edges are drawn jagged.
        """
    if strand == 1:
        inner_radius = center
        outer_radius = top
        teeth = 2
    elif strand == -1:
        inner_radius = bottom
        outer_radius = center
        teeth = 2
    else:
        inner_radius = bottom
        outer_radius = top
        teeth = 4
    tail_length_ratio = 1.0
    head_length_ratio = 1.0
    strokecolor, color = _stroke_and_fill_colors(color, border)
    startangle, endangle = (min(startangle, endangle), max(startangle, endangle))
    angle = endangle - startangle
    height = outer_radius - inner_radius
    assert startangle <= endangle and angle >= 0
    if head_length_ratio and tail_length_ratio:
        headangle = max(endangle - min(height * head_length_ratio / (center * teeth), angle * 0.5), startangle)
        tailangle = min(startangle + min(height * tail_length_ratio / (center * teeth), angle * 0.5), endangle)
        tailangle = min(tailangle, headangle)
    elif head_length_ratio:
        headangle = max(endangle - min(height * head_length_ratio / (center * teeth), angle), startangle)
        tailangle = startangle
    else:
        headangle = endangle
        tailangle = min(startangle + min(height * tail_length_ratio / (center * teeth), angle), endangle)
    if not startangle <= tailangle <= headangle <= endangle:
        raise RuntimeError('Problem drawing jaggy sigil, invalid positions. Start angle: %s, Tail angle: %s, Head angle: %s, End angle %s, Angle: %s' % (startangle, tailangle, headangle, endangle, angle))
    startcos, startsin = (cos(startangle), sin(startangle))
    headcos, headsin = (cos(headangle), sin(headangle))
    endcos, endsin = (cos(endangle), sin(endangle))
    x0, y0 = (self.xcenter, self.ycenter)
    p = ArcPath(strokeColor=strokecolor, fillColor=color, strokeLineJoin=1, strokewidth=0, **kwargs)
    p.addArc(self.xcenter, self.ycenter, inner_radius, 90 - headangle * 180 / pi, 90 - tailangle * 180 / pi, moveTo=True)
    for i in range(teeth):
        p.addArc(self.xcenter, self.ycenter, inner_radius + i * height / teeth, 90 - tailangle * 180 / pi, 90 - startangle * 180 / pi)
        self._draw_arc_line(p, inner_radius + i * height / teeth, inner_radius + (i + 1) * height / teeth, 90 - startangle * 180 / pi, 90 - tailangle * 180 / pi)
    p.addArc(self.xcenter, self.ycenter, outer_radius, 90 - headangle * 180 / pi, 90 - tailangle * 180 / pi, reverse=True)
    for i in range(teeth):
        p.addArc(self.xcenter, self.ycenter, outer_radius - i * height / teeth, 90 - endangle * 180 / pi, 90 - headangle * 180 / pi, reverse=True)
        self._draw_arc_line(p, outer_radius - i * height / teeth, outer_radius - (i + 1) * height / teeth, 90 - endangle * 180 / pi, 90 - headangle * 180 / pi)
    p.closePath()
    return p