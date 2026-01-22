from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_header_application(self):
    """BLAST program, e.g., blastp, blastn, etc. (PRIVATE).

        Save this to put on each blast record object
        """
    self._header.application = self._value.upper()