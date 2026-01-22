import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def _clade_to_bitstr(clade, tree_term_names):
    """Create a BitString representing a clade, given ordered tree taxon names (PRIVATE)."""
    clade_term_names = {term.name for term in clade.find_clades(terminal=True)}
    return _BitString.from_bool((name in clade_term_names for name in tree_term_names))