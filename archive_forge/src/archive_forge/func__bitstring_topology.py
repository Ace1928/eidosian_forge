import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def _bitstring_topology(tree):
    """Generate a branch length dict for a tree, keyed by BitStrings (PRIVATE).

    Create a dict of all clades' BitStrings to the corresponding branch
    lengths (rounded to 5 decimal places).
    """
    bitstrs = {}
    for clade, bitstr in _tree_to_bitstrs(tree).items():
        bitstrs[bitstr] = round(clade.branch_length or 0.0, 5)
    return bitstrs