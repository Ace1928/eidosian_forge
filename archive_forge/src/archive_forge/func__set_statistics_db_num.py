from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_statistics_db_num(self):
    """Record the number of sequences in the database (PRIVATE)."""
    self._blast.num_sequences_in_database = int(self._value)