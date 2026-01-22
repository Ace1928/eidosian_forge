from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_positive(self):
    """Record the number of positive (conservative) substitutions in the alignment (PRIVATE)."""
    self._hsp.positives = int(self._value)