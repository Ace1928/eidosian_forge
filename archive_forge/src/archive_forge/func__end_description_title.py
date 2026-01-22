from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _end_description_title(self):
    """XML v2. The hit description title (PRIVATE)."""
    self._hit_descr_item.title = self._value