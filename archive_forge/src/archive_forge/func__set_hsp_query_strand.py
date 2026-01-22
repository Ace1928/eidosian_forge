from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_query_strand(self):
    """Frame of the query if applicable (PRIVATE)."""
    self._hsp.strand = (self._value,)
    if self._header.application == 'BLASTN':
        self._hsp.frame = (1 if self._value == 'Plus' else -1,)