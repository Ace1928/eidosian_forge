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
def _drawLines(self):
    ccap, cdash, cjoin = (None, None, None)
    canv = self.canv
    canv.saveState()
    rrd = self._roundingRectDef
    if rrd:
        SL = rrd.SL
        SL[:] = []
        ocanvline = canv.line
        aSL = SL.append

        def rcCanvLine(xs, ys, xe, ye):
            if xs == xe and (xs >= rrd.x1 or xs <= rrd.x0) or (ys == ye and (ys >= rrd.y1 or ys <= rrd.y0)):
                aSL(RoundingRectLine(xs, ys, xe, ye, weight, color, cap, dash, join))
            else:
                ocanvline(xs, ys, xe, ye)
        canv.line = rcCanvLine
    try:
        for op, (sc, sr), (ec, er), weight, color, cap, dash, join, count, space in self._linecmds:
            if isinstance(sr, strTypes) and sr in _SPECIALROWS:
                continue
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
            sc, ec, sr, er = self.normCellRange(sc, ec, sr, er)
            getattr(self, _LineOpMap.get(op, '_drawUnknown'))((sc, sr), (ec, er), weight, color, count, space)
    finally:
        if rrd:
            canv.line = ocanvline
    canv.restoreState()
    self._curcolor = None