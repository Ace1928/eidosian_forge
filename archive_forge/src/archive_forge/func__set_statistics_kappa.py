from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_statistics_kappa(self):
    """Karlin-Altschul parameter K (PRIVATE)."""
    self._blast.ka_params = float(self._value)