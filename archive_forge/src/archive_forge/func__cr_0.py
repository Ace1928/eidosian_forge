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
def _cr_0(self, n, cmds, nr0, doInRowSplit, _srflMode=False):
    ncols = self._ncols
    for c in cmds:
        (sc, sr), (ec, er) = c[1:3]
        if sr in _SPECIALROWS:
            if sr[0] == 'i':
                self._addCommand(c)
                if sr == 'inrowsplitstart' and doInRowSplit:
                    if sc < 0:
                        sc += ncols
                    if ec < 0:
                        ec += ncols
                    self._addCommand((c[0],) + ((sc, n - 1), (ec, n - 1)) + tuple(c[3:]))
                continue
            if not _srflMode:
                continue
            self._addCommand(c)
            if sr == 'splitfirst':
                continue
            sr = er = n - 1
        if sr < 0:
            sr += nr0
        if sr >= n:
            continue
        if er >= n:
            er = n - 1
        self._addCommand((c[0],) + ((sc, sr), (ec, er)) + tuple(c[3:]))