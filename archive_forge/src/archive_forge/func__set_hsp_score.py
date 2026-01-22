from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_score(self):
    """Record the raw score of HSP (PRIVATE)."""
    self._hsp.score = float(self._value)
    if self._descr.score is None:
        self._descr.score = float(self._value)