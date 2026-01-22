import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
def _get_taxonomy(self):
    """Get taxonomy list for the clade (PRIVATE)."""
    if len(self.taxonomies) == 0:
        return None
    if len(self.taxonomies) > 1:
        raise AttributeError('more than 1 taxonomy value available; use Clade.taxonomies')
    return self.taxonomies[0]