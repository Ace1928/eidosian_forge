from collections import namedtuple
from math import floor, ceil
def _drawhorzline(surf, color, x_from, in_y, x_to):
    if x_from == x_to:
        surf.set_at((x_from, in_y), color)
        return
    start, end = (x_from, x_to) if x_from <= x_to else (x_to, x_from)
    for line_x in range(start, end + 1):
        surf.set_at((line_x, in_y), color)