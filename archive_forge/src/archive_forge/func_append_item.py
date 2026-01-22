from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
def append_item(self, item):
    """Add a description extended record."""
    if len(self.items) == 0:
        self.title = str(item)
    self.items.append(item)