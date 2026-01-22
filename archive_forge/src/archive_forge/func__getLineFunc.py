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
def _getLineFunc(self, start, end, parent=None):
    _3d_dx = getattr(parent, '_3d_dx', None)
    if _3d_dx is not None:
        _3d_dy = getattr(parent, '_3d_dy', None)
        f = self.isYAxis and self._cyLine3d or self._cxLine3d
        return lambda v, s=start, e=end, f=f, _3d_dx=_3d_dx, _3d_dy=_3d_dy: f(v, s, e, _3d_dx=_3d_dx, _3d_dy=_3d_dy)
    else:
        f = self.isYAxis and self._cyLine or self._cxLine
        return lambda v, s=start, e=end, f=f: f(v, s, e)