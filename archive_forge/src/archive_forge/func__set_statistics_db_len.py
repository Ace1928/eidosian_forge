from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_statistics_db_len(self):
    """Record the number of letters in the database (PRIVATE)."""
    self._blast.num_letters_in_database = int(self._value)