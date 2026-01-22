from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_parameters_matrix(self):
    """Matrix used (-M on legacy BLAST) (PRIVATE)."""
    self._parameters.matrix = self._value