import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
def _get_confidence(self):
    """Return confidence values (PRIVATE)."""
    if len(self.confidences) == 0:
        return None
    if len(self.confidences) > 1:
        raise AttributeError('more than 1 confidence value available; use Clade.confidences')
    return self.confidences[0]