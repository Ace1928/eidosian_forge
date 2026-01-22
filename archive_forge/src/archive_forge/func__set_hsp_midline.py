from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_midline(self):
    """Record the middle line as normally seen in BLAST report (PRIVATE)."""
    self._hsp.match = self._value
    assert len(self._hsp.match) == len(self._hsp.query)
    assert len(self._hsp.match) == len(self._hsp.sbjct)