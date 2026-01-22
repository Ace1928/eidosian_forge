import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
def _set_zero_diagonal(self):
    """Set all diagonal elements to zero (PRIVATE)."""
    for i in range(0, len(self)):
        self.matrix[i][i] = 0