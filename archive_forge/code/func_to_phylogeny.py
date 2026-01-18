import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
def to_phylogeny(self, **kwargs):
    """Create a new phylogeny containing just this clade."""
    phy = Phylogeny(root=self, date=self.date)
    phy.__dict__.update(kwargs)
    return phy