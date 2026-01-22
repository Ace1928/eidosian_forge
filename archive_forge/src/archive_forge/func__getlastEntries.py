from reportlab.lib.units import cm
from reportlab.lib.utils import commasplit, escapeOnce, encode_label, decode_label, strTypes, asUnicode, asNative
from reportlab.lib.styles import ParagraphStyle, _baseFontName
from reportlab.lib import sequencer as rl_sequencer
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.doctemplate import IndexingFlowable
from reportlab.platypus.tables import TableStyle, Table
from reportlab.platypus.flowables import Spacer
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
import unicodedata
from ast import literal_eval
def _getlastEntries(self, dummy=[(['Placeholder for index'], enumerate((None,) * 3))]):
    """Return the last run's entries!  If there are none, returns dummy."""
    lE = self._lastEntries or self._entries
    if not lE:
        return dummy
    return list(sorted(lE.items()))