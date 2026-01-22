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
def _splitAnnotations(self):
    A = getattr(self, 'annotations', [])[:]
    D = {}
    for v in ('early', 'beforeAxis', 'afterAxis', 'beforeTicks', 'afterTicks', 'beforeTickLabels', 'afterTickLabels', 'late'):
        R = [].append
        P = [].append
        for a in A:
            if getattr(a, v, 0):
                R(a)
            else:
                P(a)
        D[v] = R.__self__
        A[:] = P.__self__
    D['late'] += A
    return D