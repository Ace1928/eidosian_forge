from reportlab.graphics.shapes import Drawing, Line, String, Group, Polygon
from reportlab.lib import colors
from ._AbstractDrawer import AbstractDrawer, draw_box, draw_arrow
from ._AbstractDrawer import draw_cut_corner_box, _stroke_and_fill_colors
from ._AbstractDrawer import intermediate_points, angle2trig, deduplicate
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import ceil
Draw BIGARROW sigil, like ARROW but straddles the axis (PRIVATE).