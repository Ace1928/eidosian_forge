from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_header_reference(self):
    """Record any article reference describing the algorithm (PRIVATE).

        Save this to put on each blast record object
        """
    self._header.reference = self._value