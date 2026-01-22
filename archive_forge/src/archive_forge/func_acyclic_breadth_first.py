import inspect
import locale
import os
import pydoc
import re
import textwrap
import warnings
from collections import defaultdict, deque
from itertools import chain, combinations, islice, tee
from pprint import pprint
from urllib.request import (
from nltk.collections import *
from nltk.internals import deprecated, raise_unorderable_types, slice_bounds
def acyclic_breadth_first(tree, children=iter, maxdepth=-1):
    """Traverse the nodes of a tree in breadth-first order,
    discarding eventual cycles.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.
    """
    traversed = set()
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        yield node
        traversed.add(node)
        if depth != maxdepth:
            try:
                for child in children(node):
                    if child not in traversed:
                        queue.append((child, depth + 1))
                    else:
                        warnings.warn('Discarded redundant search for {} at depth {}'.format(child, depth + 1), stacklevel=2)
            except TypeError:
                pass