from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_parameters_filter(self):
    """Record filtering options (-F) (PRIVATE)."""
    self._parameters.filter = self._value