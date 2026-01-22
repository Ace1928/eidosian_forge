from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
def _ystrip_poly(x0, x1, y0, y1, xoff, yoff):
    return [x0, y0, x0 + xoff, y0 + yoff, x1 + xoff, y1 + yoff, x1, y1]