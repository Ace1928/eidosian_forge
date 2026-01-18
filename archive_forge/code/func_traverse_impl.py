from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def traverse_impl(path, structure):
    """Recursive traversal implementation."""

    def subtree_fn(item):
        subtree_path, subtree = item
        return traverse_impl(path + (subtree_path,), subtree)

    def traverse_subtrees():
        if is_nested(structure):
            return _sequence_like(structure, map(subtree_fn, _yield_sorted_items(structure)))
        else:
            return structure
    if top_down:
        ret = fn(path, structure)
        if ret is None:
            return traverse_subtrees()
        elif ret is MAP_TO_NONE:
            return None
        else:
            return ret
    else:
        traversed_structure = traverse_subtrees()
        ret = fn(path, traversed_structure)
        if ret is None:
            return traversed_structure
        elif ret is MAP_TO_NONE:
            return None
        else:
            return ret