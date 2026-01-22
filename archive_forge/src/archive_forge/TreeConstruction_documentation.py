import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
Build the tree.

        :Parameters:
            alignment : MultipleSeqAlignment
                multiple sequence alignment to calculate parsimony tree.

        