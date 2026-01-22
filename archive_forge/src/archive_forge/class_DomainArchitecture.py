import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class DomainArchitecture(PhyloElement):
    """Domain architecture of a protein.

    :Parameters:
        length : int
            total length of the protein sequence
        domains : list ProteinDomain objects
            the domains within this protein

    """

    def __init__(self, length=None, domains=None):
        """Initialize values of the DomainArchitecture object."""
        self.length = length
        self.domains = domains