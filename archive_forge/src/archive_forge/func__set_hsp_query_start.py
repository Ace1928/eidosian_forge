from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_query_start(self):
    """Offset of query at the start of the alignment (one-offset) (PRIVATE)."""
    self._hsp.query_start = int(self._value)