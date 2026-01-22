from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _end_description_taxid(self):
    try:
        self._hit_descr_item.taxid = int(self._value)
    except ValueError:
        pass