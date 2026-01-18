from __future__ import annotations
import difflib
import functools
import itertools
import textwrap
from collections import OrderedDict, defaultdict, deque
from typing import Any, Callable, Iterable, Mapping, overload
from optree import _C
from optree.registry import (
from optree.typing import (
from optree.typing import structseq as PyStructSequence  # noqa: N812
from optree.typing import structseq_fields
def tree_paths(tree: PyTree[T], is_leaf: Callable[[T], bool] | None=None, *, none_is_leaf: bool=False, namespace: str='') -> list[tuple[Any, ...]]:
    """Get the path entries to the leaves of a pytree.

    See also :func:`tree_flatten`, :func:`tree_flatten_with_path`, and :func:`treespec_paths`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_paths(tree)
    [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('d',)]
    >>> tree_paths(tree, none_is_leaf=True)
    [('a',), ('b', 0), ('b', 1, 0), ('b', 1, 1), ('c',), ('d',)]
    >>> tree_paths(1)
    [()]
    >>> tree_paths(None)
    []
    >>> tree_paths(None, none_is_leaf=True)
    [()]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A list of the paths to the leaf values, while each path is a tuple of the index or keys.
    """
    return _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)[0]