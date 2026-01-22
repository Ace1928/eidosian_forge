from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_record_query_def(self):
    """Record the definition line of the query (PRIVATE)."""
    self._blast.query = self._value