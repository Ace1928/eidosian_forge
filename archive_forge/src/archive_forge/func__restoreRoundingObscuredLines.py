from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
def _restoreRoundingObscuredLines(self):
    x0, y0, w, h, x1, y1, ar, SL = self._roundingRectDef
    if not SL:
        return
    canv = self.canv
    canv.saveState()
    ccap = cdash = cjoin = self._curweight = self._curcolor = None
    line = canv.line
    cornerRadii = self._cornerRadii
    for xs, ys, xe, ye, weight, color, cap, dash, join in SL:
        if cap != None and ccap != cap:
            canv.setLineCap(cap)
            ccap = cap
        if dash is None or dash == []:
            if cdash is not None:
                canv.setDash()
                cdash = None
        elif dash != cdash:
            canv.setDash(dash)
            cdash = dash
        if join is not None and cjoin != join:
            canv.setLineJoin(join)
            cjoin = join
        self._prepLine(weight, color)
        if ys == ye:
            if ys > y1 or ys < y0:
                line(xs, ys, xe, ye)
                continue
            if ys == y0:
                ypos = 'bottom'
                r0 = ar[2]
                r1 = ar[3]
            else:
                ypos = 'top'
                r0 = ar[0]
                r1 = ar[1]
            if xs >= x0 + r0 and xe <= x1 - r1:
                line(xs, ys, xe, ye)
                continue
            c0 = _quadrantDef('left', ypos, (xs, ys), r0, kind=2, direction='left-right') if xs < x0 + r0 else None
            c1 = _quadrantDef('right', ypos, (xe, ye), r1, kind=1, direction='left-right') if xe > x1 - r1 else None
        else:
            if xs > x1 or xs < x0:
                line(xs, ys, xe, ye)
                continue
            if xs == x0:
                xpos = 'left'
                r0 = ar[2]
                r1 = ar[0]
            else:
                xpos = 'right'
                r0 = ar[3]
                r1 = ar[1]
            if ys >= y0 + r0 and ye <= y1 - r1:
                line(xs, ys, xe, ye)
                continue
            c0 = _quadrantDef(xpos, 'bottom', (xs, ys), r0, kind=2, direction='bottom-top') if ys < y0 + r0 else None
            c1 = _quadrantDef(xpos, 'top', (xe, ye), r1, kind=1, direction='bottom-top') if ye > y1 - r1 else None
        P = canv.beginPath()
        if c0:
            P.moveTo(*c0[0])
            P.curveTo(c0[1][0], c0[1][1], c0[2][0], c0[2][1], c0[3][0], c0[3][1])
        else:
            P.moveTo(xs, ys)
        if not c1:
            P.lineTo(xe, ye)
        else:
            P.lineTo(*c1[0])
            P.curveTo(c1[1][0], c1[1][1], c1[2][0], c1[2][1], c1[3][0], c1[3][1])
        canv.drawPath(P, stroke=1, fill=0)
    canv.restoreState()