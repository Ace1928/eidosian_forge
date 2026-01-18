from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
def ungap(self, gap='-'):
    """Return a copy of the sequence without the gap character(s)."""
    if len(gap) != 1 or not isinstance(gap, str):
        raise ValueError(f'Unexpected gap character, {gap!r}')
    return CodonSeq(str(self).replace(gap, ''), rf_table=self.rf_table)