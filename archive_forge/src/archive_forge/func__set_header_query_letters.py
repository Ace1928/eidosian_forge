from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_header_query_letters(self):
    """Record the length of the query (PRIVATE).

        Important in old pre 2.2.14 BLAST, for recent versions
        <Iteration_query-len> is enough
        """
    self._header.query_letters = int(self._value)