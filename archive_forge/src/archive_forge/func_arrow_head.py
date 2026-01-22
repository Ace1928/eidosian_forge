import tkinter
from tkinter.constants import *
def arrow_head(document, x0, y0, x1, y1, arrowshape):
    """make arrow head at (x1,y1), arrowshape is tuple (d1, d2, d3)"""
    import math
    dx = x1 - x0
    dy = y1 - y0
    poly = document.createElement('polygon')
    d = math.sqrt(dx * dx + dy * dy)
    if d == 0.0:
        return poly
    try:
        d1, d2, d3 = list(map(float, arrowshape))
    except ValueError:
        d1, d2, d3 = map(float, arrowshape.split())
    P0 = (x0, y0)
    P1 = (x1, y1)
    xa, ya = lerp(P1, P0, d1 / d)
    xb, yb = lerp(P1, P0, d2 / d)
    t = d3 / d
    xc, yc = (dx * t, dy * t)
    points = [x1, y1, xb - yc, yb + xc, xa, ya, xb + yc, yb - xc]
    poly.setAttribute('points', ' '.join(map(str, points)))
    return poly