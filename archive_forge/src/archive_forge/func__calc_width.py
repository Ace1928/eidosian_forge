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
def _calc_width(self, availWidth, W=None):
    if getattr(self, '_width_calculated_once', None):
        return
    if not W:
        W = _calc_pc(self._argW, availWidth)
    if None in W:
        canv = getattr(self, 'canv', None)
        saved = None
        if self._spanCmds:
            colSpanCells = self._colSpanCells
            spanRanges = self._spanRanges
        else:
            colSpanCells = ()
            spanRanges = {}
        spanCons = {}
        if W is self._argW:
            W0 = W
            W = W[:]
        else:
            W0 = W[:]
        V = self._cellvalues
        S = self._cellStyles
        while None in W:
            j = W.index(None)
            w = 0
            for i, Vi in enumerate(V):
                v = Vi[j]
                s = S[i][j]
                ji = (j, i)
                span = spanRanges.get(ji, None)
                if ji in colSpanCells and (not span):
                    t = 0.0
                else:
                    t = self._elementWidth(v, s)
                    if t is None:
                        raise ValueError(f"Flowable {v.identity()} in cell({i},{j}) can't have auto width\n{self.identity(30)}")
                    t += s.leftPadding + s.rightPadding
                    if span:
                        c0 = span[0]
                        c1 = span[2]
                        if c0 != c1:
                            x = (c0, c1)
                            spanCons[x] = max(spanCons.get(x, t), t)
                            t = 0
                if t > w:
                    w = t
            W[j] = w
        if spanCons:
            try:
                spanFixDim(W0, W, spanCons)
            except:
                annotateException('\nspanning problem in %s\nW0=%r W=%r\nspanCons=%r' % (self.identity(), W0, W, spanCons))
    self._colWidths = W
    width = 0
    self._colpositions = [0]
    for w in W:
        width = width + w
        self._colpositions.append(width)
    self._width = width
    self._width_calculated_once = 1