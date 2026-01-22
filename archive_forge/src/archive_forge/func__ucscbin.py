import os
from itertools import islice
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequentialAlignmentWriter
@staticmethod
def _ucscbin(start, end):
    """Return the smallest bin a given region will fit into (PRIVATE).

        Adapted from http://genomewiki.ucsc.edu/index.php/Bin_indexing_system
        """
    bin_offsets = [512 + 64 + 8 + 1, 64 + 8 + 1, 8 + 1, 1, 0]
    _bin_first_shift = 17
    _bin_next_shift = 3
    start_bin = start
    end_bin = end - 1
    start_bin >>= _bin_first_shift
    end_bin >>= _bin_first_shift
    for bin_offset in bin_offsets:
        if start_bin == end_bin:
            return bin_offset + start_bin
        start_bin >>= _bin_next_shift
        end_bin >>= _bin_next_shift
    return 0