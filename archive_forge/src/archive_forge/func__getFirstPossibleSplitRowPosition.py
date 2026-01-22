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
def _getFirstPossibleSplitRowPosition(self, availHeight, ignoreSpans=0):
    impossible = {}
    if self._spanCmds and (not ignoreSpans):
        self._getRowImpossible(impossible, self._rowSpanCells, self._spanRanges)
    if self._nosplitCmds:
        self._getRowImpossible(impossible, self._rowNoSplitCells, self._nosplitRanges)
    h = 0
    n = 1
    split_at = 0
    for rh in self._rowHeights:
        if h + rh > availHeight:
            break
        if n not in impossible:
            split_at = n
        h = h + rh
        n = n + 1
    return split_at