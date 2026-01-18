import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def tree_leaves(tree: PyTree) -> List[Any]:
    """Get the leaves of a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_leaves(tree)
    [1, 2, 3, 4, None, 5]
    >>> tree_leaves(1)
    [1]
    >>> tree_leaves(None)
    [None]

    Args:
        tree (pytree): A pytree to flatten.

    Returns:
        A list of leaf values.
    """
    return optree.tree_leaves(tree, none_is_leaf=True, namespace='torch')