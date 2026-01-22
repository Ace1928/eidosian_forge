from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def _draw_arc_line(self, path, start_radius, end_radius, start_angle, end_angle, move=False):
    """Add a list of points to a path object (PRIVATE).

        Assumes angles given are in degrees!

        Represents what would be a straight line on a linear diagram.
        """
    x0, y0 = (self.xcenter, self.ycenter)
    radius_diff = end_radius - start_radius
    angle_diff = end_angle - start_angle
    dx = 0.01
    a = start_angle * pi / 180
    if move:
        path.moveTo(x0 + start_radius * cos(a), y0 + start_radius * sin(a))
    else:
        path.lineTo(x0 + start_radius * cos(a), y0 + start_radius * sin(a))
    x = dx
    if 0.01 <= abs(dx):
        while x < 1:
            r = start_radius + x * radius_diff
            a = (start_angle + x * angle_diff) * pi / 180
            path.lineTo(x0 + r * cos(a), y0 + r * sin(a))
            x += dx
    a = end_angle * pi / 180
    path.lineTo(x0 + end_radius * cos(a), y0 + end_radius * sin(a))