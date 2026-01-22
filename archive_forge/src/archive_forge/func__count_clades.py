import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def _count_clades(trees):
    """Count distinct clades (different sets of terminal names) in the trees (PRIVATE).

    Return a tuple first a dict of bitstring (representing clade) and a tuple of its count of
    occurrences and sum of branch length for that clade, second the number of trees processed.

    :Parameters:
        trees : iterable
            An iterable that returns the trees to count

    """
    bitstrs = {}
    tree_count = 0
    for tree in trees:
        tree_count += 1
        clade_bitstrs = _tree_to_bitstrs(tree)
        for clade in tree.find_clades(terminal=False):
            bitstr = clade_bitstrs[clade]
            if bitstr in bitstrs:
                count, sum_bl = bitstrs[bitstr]
                count += 1
                sum_bl += clade.branch_length or 0
                bitstrs[bitstr] = (count, sum_bl)
            else:
                bitstrs[bitstr] = (1, clade.branch_length or 0)
    return (bitstrs, tree_count)