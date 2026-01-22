from math import log10 as math_log10
from reportlab.lib.validators import    isNumber, isNumberOrNone, isListOfStringsOrNone, isListOfNumbers, \
from reportlab.lib.attrmap import *
from reportlab.lib import normalDate
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, Group, STATE_DEFAULTS, _textBoxLimits, _rotatedBoxLimits
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection
from reportlab.graphics.charts.textlabels import Label, PMVLabel, XLabel,  DirectDrawFlowable
from reportlab.graphics.charts.utils import nextRoundNumber
from reportlab.graphics.widgets.grids import ShadedRect
from reportlab.lib.colors import Color
from reportlab.lib.utils import isSeq
def _findMinMaxValue(V, x, default, func, special=None, extraMinMaxValues=None):
    if isSeq(V[0][0]):
        if special:
            f = lambda T, x=x, special=special, func=func: special(T, x, func)
        else:
            f = lambda T, x=x: T[x]
        V = list(map(lambda e, f=f: list(map(f, e)), V))
    V = list(filter(len, [[x for x in x if x is not None] for x in V]))
    if len(V) == 0:
        return default
    r = func(list(map(func, V)))
    return func(func(extraMinMaxValues), r) if extraMinMaxValues else r