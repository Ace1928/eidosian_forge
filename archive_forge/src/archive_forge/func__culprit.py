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
def _culprit(self):
    """Return a string describing the tallest element.

        Usually this is what causes tables to fail to split.  Currently
        tables are the only items to have a '_culprit' method. Doctemplate
        checks for it.
        """
    rh = self._rowHeights
    tallest = max(rh)
    rowNum = rh.index(tallest)
    return 'tallest cell %0.1f points' % tallest