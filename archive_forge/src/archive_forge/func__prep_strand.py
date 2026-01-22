import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _prep_strand(self, strand):
    if strand not in (-1, 0, 1, None):
        raise ValueError('Strand should be -1, 0, 1, or None; not %r' % strand)
    return strand