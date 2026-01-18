from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def ticks(lower, upper, n=(4, 5, 6, 7, 8, 9), split=1, percent=0, grid=None, labelVOffset=0):
    """
    return tick positions and labels for range lower<=x<=upper
    n=number of intervals to try (can be a list or sequence)
    split=1 return ticks then labels else (tick,label) pairs
    """
    t, hi, grid = find_good_grid(lower, upper, n, grid)
    power = floor(log10(grid))
    if power == 0:
        power = 1
    w = grid / 10.0 ** power
    w = int(w) != w
    if power > 3 or power < -3:
        format = '%+' + repr(w + 7) + '.0e'
    elif power >= 0:
        digits = int(power) + w
        format = '%' + repr(digits) + '.0f'
    else:
        digits = w - int(power)
        format = '%' + repr(digits + 2) + '.' + repr(digits) + 'f'
    if percent:
        format = format + '%%'
    T = []
    n = int(float(hi - t) / grid + 0.1) + 1
    if split:
        labels = []
        for i in range(n):
            v = t + grid * i
            T.append(v)
            labels.append(format % (v + labelVOffset))
        return (T, labels)
    else:
        for i in range(n):
            v = t + grid * i
            T.append((v, format % (v + labelVOffset)))
        return T