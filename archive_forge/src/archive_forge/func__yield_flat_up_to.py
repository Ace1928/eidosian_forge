from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def _yield_flat_up_to(shallow_tree, input_tree, path=()):
    """Yields (path, value) pairs of input_tree flattened up to shallow_tree.

  Args:
    shallow_tree: Nested structure. Traverse no further than its leaf nodes.
    input_tree: Nested structure. Return the paths and values from this tree.
      Must have the same upper structure as shallow_tree.
    path: Tuple. Optional argument, only used when recursing. The path from the
      root of the original shallow_tree, down to the root of the shallow_tree
      arg of this recursive call.

  Yields:
    Pairs of (path, value), where path the tuple path of a leaf node in
    shallow_tree, and value is the value of the corresponding node in
    input_tree.
  """
    if isinstance(shallow_tree, _TEXT_OR_BYTES) or not (isinstance(shallow_tree, (collections_abc.Mapping, collections_abc.Sequence)) or _is_namedtuple(shallow_tree) or _is_attrs(shallow_tree)):
        yield (path, input_tree)
    else:
        input_tree = dict(_yield_sorted_items(input_tree))
        for shallow_key, shallow_subtree in _yield_sorted_items(shallow_tree):
            subpath = path + (shallow_key,)
            input_subtree = input_tree[shallow_key]
            for leaf_path, leaf_value in _yield_flat_up_to(shallow_subtree, input_subtree, path=subpath):
                yield (leaf_path, leaf_value)