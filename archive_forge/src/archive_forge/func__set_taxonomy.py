import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
def _set_taxonomy(self, value):
    """Set a taxonomy for the clade (PRIVATE)."""
    if not isinstance(value, Taxonomy):
        raise ValueError('assigned value must be a Taxonomy instance')
    if len(self.taxonomies) == 0:
        self.taxonomies.append(value)
    elif len(self.taxonomies) == 1:
        self.taxonomies[0] = value
    else:
        raise ValueError('multiple taxonomy values already exist; use Phylogeny.taxonomies instead')