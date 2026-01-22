from collections import namedtuple
from math import floor, ceil
def _drawvertline(surf, color, in_x, y_from, y_to):
    if y_from == y_to:
        surf.set_at((in_x, y_from), color)
        return
    start, end = (y_from, y_to) if y_from <= y_to else (y_to, y_from)
    for line_y in range(start, end + 1):
        surf.set_at((in_x, line_y), color)