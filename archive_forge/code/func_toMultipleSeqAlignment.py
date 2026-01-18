from math import sqrt, erfc
import warnings
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio import BiopythonWarning
from Bio.codonalign.codonseq import _get_codon_list, CodonSeq, cal_dn_ds
def toMultipleSeqAlignment(self):
    """Convert the CodonAlignment to a MultipleSeqAlignment.

        Return a MultipleSeqAlignment containing all the
        SeqRecord in the CodonAlignment using Seq to store
        sequences
        """
    alignments = [SeqRecord(rec.seq.toSeq(), id=rec.id) for rec in self._records]
    return MultipleSeqAlignment(alignments)