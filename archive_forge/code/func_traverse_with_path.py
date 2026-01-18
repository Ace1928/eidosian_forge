from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def traverse_with_path(fn, structure, top_down=True):
    """Traverses the given nested structure, applying the given function.

  The traversal is depth-first. If ``top_down`` is True (default), parents
  are returned before their children (giving the option to avoid traversing
  into a sub-tree).

  >>> visited = []
  >>> tree.traverse_with_path(
  ...  lambda path, subtree: visited.append((path, subtree)),
  ...  [(1, 2), [3], {"a": 4}],
  ...  top_down=True)
  [(1, 2), [3], {'a': 4}]
  >>> visited == [
  ...  ((), [(1, 2), [3], {'a': 4}]),
  ...  ((0,), (1, 2)),
  ...  ((0, 0), 1),
  ...  ((0, 1), 2),
  ...  ((1,), [3]),
  ...  ((1, 0), 3),
  ...  ((2,), {'a': 4}),
  ...  ((2, 'a'), 4)]
  True

  >>> visited = []
  >>> tree.traverse_with_path(
  ...  lambda path, subtree: visited.append((path, subtree)),
  ...  [(1, 2), [3], {"a": 4}],
  ...  top_down=False)
  [(1, 2), [3], {'a': 4}]
  >>> visited == [
  ...  ((0, 0), 1),
  ...  ((0, 1), 2),
  ...  ((0,), (1, 2)),
  ...  ((1, 0), 3),
  ...  ((1,), [3]),
  ...  ((2, 'a'), 4),
  ...  ((2,), {'a': 4}),
  ... ((), [(1, 2), [3], {'a': 4}])]
  True

  Args:
    fn: The function to be applied to the path to each sub-nest of the structure
      and the sub-nest value.
      When traversing top-down: If ``fn(path, subtree) is None`` the traversal
        continues into the sub-tree. If ``fn(path, subtree) is not None`` the
        traversal does not continue into the sub-tree. The sub-tree will be
        replaced by ``fn(path, subtree)`` in the returned structure (to replace
        the sub-tree with None, use the special
        value :data:`MAP_TO_NONE`).
      When traversing bottom-up: If ``fn(path, subtree) is None`` the traversed
        sub-tree is returned unaltered. If ``fn(path, subtree) is not None`` the
        sub-tree will be replaced by ``fn(path, subtree)`` in the returned
        structure (to replace the sub-tree
        with None, use the special value :data:`MAP_TO_NONE`).
    structure: The structure to traverse.
    top_down: If True, parent structures will be visited before their children.

  Returns:
    The structured output from the traversal.
  """

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
    return traverse_impl((), structure)