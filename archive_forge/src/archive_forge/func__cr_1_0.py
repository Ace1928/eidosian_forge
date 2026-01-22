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
def _cr_1_0(self, n, cmds, doInRowSplit, _srflMode=False):
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
                    self._addCommand((c[0],) + ((sc, 0), (ec, 0)) + tuple(c[3:]))
                continue
            if not _srflMode:
                continue
            self._addCommand(c)
            if sr == 'splitlast':
                continue
            sr = er = n
        if er >= 0 and er < n:
            continue
        if sr >= 0 and sr < n:
            sr = 0
        if sr >= n:
            sr -= n
        if er >= n:
            er -= n
        self._addCommand((c[0],) + ((sc, sr), (ec, er)) + tuple(c[3:]))