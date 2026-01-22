from collections import namedtuple
from math import floor, ceil
def _draw_aaline_dy(d_y, slope, end, start, draw_two_pixel):
    g_y = ceil(start.y)
    g_x = start.x + (g_y - start.y) * slope
    if start.y < g_y:
        draw_two_pixel(g_x - slope, floor(start.y), inv_frac(start.y))
    rest = frac(end.y)
    s_y = ceil(end.y)
    if rest > 0:
        s_x = start.x + slope * (d_y + 1 - rest)
        draw_two_pixel(s_x, s_y, rest)
    else:
        s_y += 1
    for line_y in range(g_y, s_y):
        line_x = g_x + slope * (line_y - g_y)
        draw_two_pixel(line_x, line_y, 1)