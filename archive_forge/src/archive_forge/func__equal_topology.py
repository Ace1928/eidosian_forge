import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def _equal_topology(tree1, tree2):
    """Are two trees are equal in terms of topology and branch lengths (PRIVATE).

    (Branch lengths checked to 5 decimal places.)
    """
    term_names1 = {term.name for term in tree1.find_clades(terminal=True)}
    term_names2 = {term.name for term in tree2.find_clades(terminal=True)}
    return term_names1 == term_names2 and _bitstring_topology(tree1) == _bitstring_topology(tree2)