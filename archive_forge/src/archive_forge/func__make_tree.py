from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
def _make_tree(sequence):
    """Recursively creates a tree from the given sequence of nested
        tuples.

        This function employs the :func:`~networkx.tree.join` function
        to recursively join subtrees into a larger tree.

        """
    if len(sequence) == 0:
        return nx.empty_graph(1)
    return nx.tree.join_trees([(_make_tree(child), 0) for child in sequence])