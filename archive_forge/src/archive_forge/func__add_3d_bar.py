from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
def _add_3d_bar(x1, x2, y1, y2, xoff, yoff, G=G, strokeColor=strokeColor, strokeWidth=strokeWidth, fillColor=fillColor):
    G.add(Polygon((x1, y1, x1 + xoff, y1 + yoff, x2 + xoff, y2 + yoff, x2, y2), strokeWidth=strokeWidth, strokeColor=strokeColor, fillColor=fillColor, strokeLineJoin=1))