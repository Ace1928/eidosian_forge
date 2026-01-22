from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_statistics_lambda(self):
    """Karlin-Altschul parameter Lambda (PRIVATE)."""
    self._blast.ka_params = (float(self._value), self._blast.ka_params)