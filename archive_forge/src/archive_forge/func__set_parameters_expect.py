from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_parameters_expect(self):
    """Expect values cutoff (PRIVATE)."""
    self._parameters.expect = self._value