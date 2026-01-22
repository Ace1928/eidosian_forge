from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_parameters_sc_match(self):
    """Match score for nucleotide-nucleotide comparison (-r) (PRIVATE)."""
    self._parameters.sc_match = int(self._value)