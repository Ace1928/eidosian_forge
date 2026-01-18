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
def tree_max(tree, *, default=__MISSING, key=None, is_leaf=None, none_is_leaf=False, namespace=''):
    """Return the maximum leaf value in ``tree``.

    See also :func:`tree_leaves` and :func:`tree_min`.

    >>> tree_max({})
    Traceback (most recent call last):
        ...
    ValueError: max() arg is an empty sequence
    >>> tree_max({}, default=0)
    0
    >>> tree_max({'x': 0, 'y': (2, 1)})
    2
    >>> tree_max({'x': 0, 'y': (2, 1)}, key=lambda x: -x)
    0
    >>> tree_max({'a': None})  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: max() arg is an empty sequence
    >>> tree_max({'a': None}, default=0)  # `None` is a non-leaf node with arity 0 by default
    0
    >>> tree_max({'a': None}, none_is_leaf=True)
    None
    >>> tree_max(None)  # `None` is a non-leaf node with arity 0 by default
    Traceback (most recent call last):
        ...
    ValueError: max() arg is an empty sequence
    >>> tree_max(None, default=0)
    0
    >>> tree_max(None, none_is_leaf=True)
    None

    Args:
        tree (pytree): A pytree to be traversed.
        default (object, optional): The default value to return if ``tree`` is empty. If the ``tree``
            is empty and ``default`` is not specified, raise a :exc:`ValueError`.
        key (callable or None, optional): An one argument ordering function like that used for
            :meth:`list.sort`.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        The maximum leaf value in ``tree``.
    """
    leaves = tree_leaves(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    if default is __MISSING:
        if key is None:
            return max(leaves)
        return max(leaves, key=key)
    if key is None:
        return max(leaves, default=default)
    return max(leaves, default=default, key=key)