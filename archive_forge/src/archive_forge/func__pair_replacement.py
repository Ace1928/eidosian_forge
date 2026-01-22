import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def _pair_replacement(self, seq1, seq2, weight1, weight2, dictionary, letters):
    """Compare two sequences and generate info on the replacements seen (PRIVATE).

        Arguments:
         - seq1, seq2 - The two sequences to compare.
         - weight1, weight2 - The relative weights of seq1 and seq2.
         - dictionary - The dictionary containing the starting replacement
           info that we will modify.
         - letters - A list of characters to include when calculating replacements.

        """
    for residue1, residue2 in zip(seq1, seq2):
        if residue1 in letters and residue2 in letters:
            dictionary[residue1, residue2] += weight1 * weight2