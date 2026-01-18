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
def tree_broadcast_prefix(prefix_tree: PyTree[T], full_tree: PyTree[S], is_leaf: Callable[[T], bool] | None=None, *, none_is_leaf: bool=False, namespace: str='') -> PyTree[T]:
    """Return a pytree of same structure of ``full_tree`` with broadcasted subtrees in ``prefix_tree``.

    See also :func:`broadcast_prefix`, :func:`tree_broadcast_common`, and :func:`treespec_is_prefix`.

    If a ``prefix_tree`` is a prefix of a ``full_tree``, this means the ``full_tree`` can be
    constructed by replacing the leaves of ``prefix_tree`` with appropriate **subtrees**.

    This function returns a pytree with the same size as ``full_tree``. The leaves are replicated
    from ``prefix_tree``. The number of replicas is determined by the corresponding subtree in
    ``full_tree``.

    >>> tree_broadcast_prefix(1, [2, 3, 4])
    [1, 1, 1]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, 6])
    [1, 2, 3]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4; list: [4, 5, 6, 7].
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, (6, 7)])
    [1, 2, (3, 3)]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}])
    [1, 2, {'a': 3, 'b': 3, 'c': (None, 3)}]
    >>> tree_broadcast_prefix([1, 2, 3], [4, 5, {'a': 6, 'b': 7, 'c': (None, 8)}], none_is_leaf=True)
    [1, 2, {'a': 3, 'b': 3, 'c': (3, 3)}]

    Args:
        prefix_tree (pytree): A pytree with the prefix structure of ``full_tree``.
        full_tree (pytree): A pytree with the suffix structure of ``prefix_tree``.
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
        A pytree of same structure of ``full_tree`` with broadcasted subtrees in ``prefix_tree``.
    """

    def broadcast_leaves(x: T, subtree: PyTree[S]) -> PyTree[T]:
        subtreespec = tree_structure(subtree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
        return subtreespec.unflatten(itertools.repeat(x, subtreespec.num_leaves))
    return tree_map(broadcast_leaves, prefix_tree, full_tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)