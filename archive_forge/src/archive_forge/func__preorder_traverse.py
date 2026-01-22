import collections
import copy
import itertools
import random
import re
import warnings
def _preorder_traverse(root, get_children):
    """Traverse a tree in depth-first pre-order (parent before children) (PRIVATE)."""

    def dfs(elem):
        yield elem
        for v in get_children(elem):
            yield from dfs(v)
    yield from dfs(root)