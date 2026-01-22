from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _end_description_id(self):
    """XML v2. The identifier of the database sequence(PRIVATE)."""
    self._hit_descr_item.id = self._value
    if not self._hit.hit_id:
        self._hit.hit_id = self._value