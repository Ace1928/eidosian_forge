from collections import namedtuple
from math import floor, ceil
def _clip_and_draw_line(surf, rect, color, pts):
    """clip the line into the rectangle and draw if needed.

    Returns true if anything has been drawn, else false."""
    if not clip_line(pts, BoundingBox(rect.x, rect.y, rect.x + rect.w - 1, rect.y + rect.h - 1)):
        return 0
    if pts[1] == pts[3]:
        _drawhorzline(surf, color, pts[0], pts[1], pts[2])
    elif pts[0] == pts[2]:
        _drawvertline(surf, color, pts[0], pts[1], pts[3])
    else:
        _draw_line(surf, color, Point(pts[0], pts[1]), Point(pts[2], pts[3]))
    return 1