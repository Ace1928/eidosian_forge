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
def tree_transpose_map_with_path(func: Callable[..., PyTree[U]], tree: PyTree[T], *rests: PyTree[S], inner_treespec: PyTreeSpec | None=None, is_leaf: Callable[[T], bool] | None=None, none_is_leaf: bool=False, namespace: str='') -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree paths to produce a new pytree with transposed structure.

    See also :func:`tree_map_with_path`, :func:`tree_transpose_map`, and :func:`tree_transpose`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    >>> tree_transpose_map_with_path(  # doctest: +IGNORE_WHITESPACE
    ...     lambda p, x: {'depth': len(p), 'value': x},
    ...     tree,
    ... )
    {
        'depth': {'b': (2, [3, 3]), 'a': 1, 'c': (2, 2)},
        'value': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    }
    >>> tree_transpose_map_with_path(  # doctest: +IGNORE_WHITESPACE
    ...     lambda p, x: {'path': p, 'value': x},
    ...     tree,
    ...     inner_treespec=tree_structure({'path': 0, 'value': 0})),
    ... )
    {
        'path': {'b': (('b', 0), [('b', 1, 0), ('b', 1, 1)]), 'a': ('a',), 'c': (('c', 0), ('c', 1))},
        'value': {'b': (2, [3, 4]), 'a': 1, 'c': (5, 6)}
    }

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the second positional
            argument and the corresponding path providing the first positional argument to function
            ``func``.
        rests (tuple of pytree): A tuple of pytrees, each of which has the same structure as
            ``tree`` or has ``tree`` as a prefix.
        inner_treespec (PyTreeSpec, optional): The treespec object representing the inner structure
            of the result pytree. If not specified, the inner structure is inferred from the result
            of the function ``func`` on the first leaf. (default: :data:`None`)
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
        A new nested pytree with the same structure as ``inner_treespec`` but with the value at each
        leaf has the same structure as ``tree``. The subtree at each leaf is given by the result of
        function ``func(p, x, *xs)`` where ``(p, x)`` are the path and value at the corresponding
        leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in ``rests``.
    """
    paths, leaves, outer_treespec = _C.flatten_with_path(tree, is_leaf, none_is_leaf, namespace)
    if outer_treespec.num_leaves == 0:
        raise ValueError(f'The outer structure must have at least one leaf. Got: {outer_treespec}.')
    flat_args = [leaves] + [outer_treespec.flatten_up_to(r) for r in rests]
    outputs = list(map(func, paths, *flat_args))
    if inner_treespec is None:
        inner_treespec = tree_structure(outputs[0], is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    if inner_treespec.num_leaves == 0:
        raise ValueError(f'The inner structure must have at least one leaf. Got: {inner_treespec}.')
    grouped = [inner_treespec.flatten_up_to(o) for o in outputs]
    transposed = zip(*grouped)
    subtrees = map(outer_treespec.unflatten, transposed)
    return inner_treespec.unflatten(subtrees)