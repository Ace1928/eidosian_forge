import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
def _set_confidence(self, value):
    """Set the confidence value (PRIVATE)."""
    if value is None:
        self.confidences = []
        return
    if isinstance(value, (float, int)):
        value = Confidence(value)
    elif not isinstance(value, Confidence):
        raise ValueError('value must be a number or Confidence instance')
    if len(self.confidences) == 0:
        self.confidences.append(value)
    elif len(self.confidences) == 1:
        self.confidences[0] = value
    else:
        raise ValueError('multiple confidence values already exist; use Phylogeny.confidences instead')