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
def _makeRoundedCornersClip(self, FUZZ=rl_config._FUZZ):
    self._roundingRectDef = None
    cornerRadii = getattr(self, '_cornerRadii', None)
    if not cornerRadii or max(cornerRadii) <= FUZZ:
        return
    nrows = self._nrows
    ncols = self._ncols
    ar = [min(self._rowHeights[i], self._colWidths[j], cornerRadii[k]) for k, (i, j) in enumerate(((0, 0), (0, ncols - 1), (nrows - 1, 0), (nrows - 1, ncols - 1)))]
    rp = self._rowpositions
    cp = self._colpositions
    x0 = cp[0]
    y0 = rp[nrows]
    x1 = cp[ncols]
    y1 = rp[0]
    w = x1 - x0
    h = y1 - y0
    self._roundingRectDef = RoundingRectDef(x0, y0, w, h, x1, y1, ar, [])
    P = self.canv.beginPath()
    P.roundRect(x0, y0, w, h, ar)
    c = self.canv
    c.addLiteral('%begin table rect clip')
    c.clipPath(P, stroke=0)
    c.addLiteral('%end table rect clip')