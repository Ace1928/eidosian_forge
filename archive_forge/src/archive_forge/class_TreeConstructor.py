import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
class TreeConstructor:
    """Base class for all tree constructor."""

    def build_tree(self, msa):
        """Caller to build the tree from an Alignment or MultipleSeqAlignment object.

        This should be implemented in subclass.
        """
        raise NotImplementedError('Method not implemented!')