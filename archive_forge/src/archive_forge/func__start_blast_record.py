from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _start_blast_record(self):
    """Start interaction (PRIVATE)."""
    self._blast = Record.Blast()