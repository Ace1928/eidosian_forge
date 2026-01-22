from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_parameters_sc_mismatch(self):
    """Mismatch penalty for nucleotide-nucleotide comparison (-r) (PRIVATE)."""
    self._parameters.sc_mismatch = int(self._value)