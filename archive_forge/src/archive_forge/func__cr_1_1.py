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
def _cr_1_1(self, n, nRows, repeatRows, cmds, doInRowSplit, _srflMode=False):
    nrr = len(repeatRows)
    rrS = set(repeatRows)
    ncols = self._ncols
    for c in cmds:
        (sc, sr), (ec, er) = c[1:3]
        if sr in _SPECIALROWS:
            if sr[0] == 'i':
                self._addCommand(c)
                if sr == 'inrowsplitend' and doInRowSplit:
                    if sc < 0:
                        sc += ncols
                    if ec < 0:
                        ec += ncols
                    self._addCommand((c[0],) + ((sc, nrr), (ec, nrr)) + tuple(c[3:]))
                continue
            if not _srflMode:
                continue
            self._addCommand(c)
            if sr == 'splitlast':
                continue
            sr = er = n
        if sr < 0:
            sr += nRows
        if er < 0:
            er += nRows
        cS = set(range(sr, er + 1)) & rrS
        if cS:
            cS = list(cS)
            self._addCommand((c[0],) + ((sc, repeatRows.index(min(cS))), (ec, repeatRows.index(max(cS)))) + tuple(c[3:]))
        if er < n:
            continue
        sr = max(sr - n, 0) + nrr
        er = max(er - n, 0) + nrr
        self._addCommand((c[0],) + ((sc, sr), (ec, er)) + tuple(c[3:]))
    sr = self._rowSplitRange
    if sr:
        sr, er = sr
        if sr < 0:
            sr += nRows
        if er < 0:
            er += nRows
        if er < n:
            self._rowSplitRange = None
        else:
            sr = max(sr - n, 0) + nrr
            er = max(er - n, 0) + nrr
            self._rowSplitRange = (sr, er)