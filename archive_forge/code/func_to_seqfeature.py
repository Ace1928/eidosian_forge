import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
def to_seqfeature(self):
    """Create a SeqFeature from the ProteinDomain Object."""
    feat = SeqFeature(location=SimpleLocation(self.start, self.end), id=self.value)
    try:
        confidence = self.confidence
    except AttributeError:
        pass
    else:
        feat.qualifiers['confidence'] = confidence
    return feat