import copy
from collections.abc import Mapping, Iterable
from Bio import BiopythonWarning
from Bio import BiopythonExperimentalWarning
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.codonalign.codonseq import CodonSeq
from Bio.codonalign.codonalignment import CodonAlignment, mktest
import warnings
def _codons2re(codons):
    """Generate regular expression based on a given list of codons (PRIVATE)."""
    reg = ''
    for i in zip(*codons):
        if len(set(i)) == 1:
            reg += ''.join(set(i))
        else:
            reg += '[' + ''.join(set(i)) + ']'
    return reg