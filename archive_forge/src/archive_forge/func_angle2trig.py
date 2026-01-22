from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
def angle2trig(theta):
    """Convert angle to a reportlab ready tuple.

    Arguments:
     - theta -  Angle in degrees, counter clockwise from horizontal

    Returns a representation of the passed angle in a format suitable
    for ReportLab rotations (i.e. cos(theta), sin(theta), -sin(theta),
    cos(theta) tuple)
    """
    c = cos(theta * pi / 180)
    s = sin(theta * pi / 180)
    return (c, s, -s, c)