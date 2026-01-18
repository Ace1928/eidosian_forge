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
def tree_transpose(outer_treespec: PyTreeSpec, inner_treespec: PyTreeSpec, tree: PyTree[T], is_leaf: Callable[[T], bool] | None=None) -> PyTree[T]:
    """Transform a tree having tree structure (outer, inner) into one having structure (inner, outer).

    See also :func:`tree_flatten`, :func:`tree_structure`, and :func:`tree_transpose_map`.

    >>> outer_treespec = tree_structure({'a': 1, 'b': 2, 'c': (3, 4)})
    >>> outer_treespec
    PyTreeSpec({'a': *, 'b': *, 'c': (*, *)})
    >>> inner_treespec = tree_structure((1, 2))
    >>> inner_treespec
    PyTreeSpec((*, *))
    >>> tree = {'a': (1, 2), 'b': (3, 4), 'c': ((5, 6), (7, 8))}
    >>> tree_transpose(outer_treespec, inner_treespec, tree)
    ({'a': 1, 'b': 3, 'c': (5, 7)}, {'a': 2, 'b': 4, 'c': (6, 8)})

    For performance reasons, this function is only checks for the number of leaves in the input
    pytree, not the structure. The result is only enumerated up to the original order of leaves in
    ``tree``, then transpose depends on the number of leaves in structure (inner, outer). The caller
    is responsible for ensuring that the input pytree has a prefix structure of ``outer_treespec``
    followed by a prefix structure of ``inner_treespec``. Otherwise, the result may be incorrect.

    >>> tree_transpose(outer_treespec, inner_treespec, list(range(1, 9)))
    ({'a': 1, 'b': 3, 'c': (5, 7)}, {'a': 2, 'b': 4, 'c': (6, 8)})

    Args:
        outer_treespec (PyTreeSpec): A treespec object representing the outer structure of the pytree.
        inner_treespec (PyTreeSpec): A treespec object representing the inner structure of the pytree.
        tree (pytree): A pytree to be transposed.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.

    Returns:
        A new pytree with the same structure as ``inner_treespec`` but with the value at each leaf
        has the same structure as ``outer_treespec``.
    """
    if outer_treespec.none_is_leaf != inner_treespec.none_is_leaf:
        raise ValueError('Tree structures must have the same none_is_leaf value.')
    outer_size = outer_treespec.num_leaves
    inner_size = inner_treespec.num_leaves
    if outer_size == 0 or inner_size == 0:
        raise ValueError('Tree structures must have at least one leaf.')
    if outer_treespec.namespace and inner_treespec.namespace and (outer_treespec.namespace != inner_treespec.namespace):
        raise ValueError(f'Tree structures must have the same namespace, got {outer_treespec.namespace!r} vs. {inner_treespec.namespace!r}.')
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf, none_is_leaf=outer_treespec.none_is_leaf, namespace=outer_treespec.namespace or inner_treespec.namespace)
    if treespec.num_leaves != outer_size * inner_size:
        expected_treespec = outer_treespec.compose(inner_treespec)
        raise TypeError(f'Tree structure mismatch; expected: {expected_treespec}, got: {treespec}.')
    grouped = [leaves[offset:offset + inner_size] for offset in range(0, outer_size * inner_size, inner_size)]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)