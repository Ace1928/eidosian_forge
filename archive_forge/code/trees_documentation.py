import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
Recursively create a trie from the given list of paths.

        `paths` is a list of paths, each of which is itself a list of
        nodes, relative to the given `root` (but not including it). This
        list of paths will be interpreted as a tree-like structure, in
        which two paths that share a prefix represent two branches of
        the tree with the same initial segment.

        `root` is the parent of the node at index 0 in each path.

        `tree` is the "accumulator", the :class:`networkx.DiGraph`
        representing the branching to which the new nodes and edges will
        be added.

        