import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
class Scorer:
    """Base class for all tree scoring methods."""

    def get_score(self, tree, alignment):
        """Caller to get the score of a tree for the given alignment.

        This should be implemented in subclass.
        """
        raise NotImplementedError('Method not implemented!')