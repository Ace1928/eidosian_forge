from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def _draw_sigil_big_arrow(self, bottom, center, top, startangle, endangle, strand, **kwargs):
    """Draw BIGARROW sigil, like ARROW but straddles the axis (PRIVATE)."""
    if strand == -1:
        orientation = 'left'
    else:
        orientation = 'right'
    return self._draw_arc_arrow(bottom, top, startangle, endangle, orientation=orientation, **kwargs)