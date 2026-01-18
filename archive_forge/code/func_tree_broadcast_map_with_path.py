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
def tree_broadcast_map_with_path(func: Callable[..., U], tree: PyTree[T], *rests: PyTree[T], is_leaf: Callable[[T], bool] | None=None, none_is_leaf: bool=False, namespace: str='') -> PyTree[U]:
    """Map a multi-input function over pytree args as well as the tree paths to produce a new pytree.

    See also :func:`tree_broadcast_map`, :func:`tree_map`, :func:`tree_map_`,
    and :func:`tree_map_with_path`.

    If only one input is provided, this function is the same as :func:`tree_map`:

    >>> tree_broadcast_map_with_path(lambda p, x: (len(p), x), {'x': 7, 'y': (42, 64)})
    {'x': (1, 7), 'y': ((2, 42), (2, 64))}
    >>> tree_broadcast_map_with_path(lambda p, x: x + len(p), {'x': 7, 'y': (42, 64), 'z': None})
    {'x': 8, 'y': (44, 66), 'z': None}
    >>> tree_broadcast_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}})
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: None}}
    >>> tree_broadcast_map_with_path(lambda p, x: p, {'x': 7, 'y': (42, 64), 'z': {1.5: None}}, none_is_leaf=True)
    {'x': ('x',), 'y': (('y', 0), ('y', 1)), 'z': {1.5: ('z', 1.5)}}

    If multiple inputs are given, all input trees will be broadcasted to the common suffix structure
    of all inputs:

    >>> tree_broadcast_map_with_path(lambda p, x, y: (p, x * y), [5, 6, (3, 4)], [{'a': 7, 'b': 9}, [1, 2], 8])
    [{'a': ((0, 'a'), 35), 'b': ((0, 'b'), 45)},
     [((1, 0), 6), ((1, 1), 12)],
     (((2, 0), 24), ((2, 1), 32))]

    Args:
        func (callable): A function that takes ``2 + len(rests)`` arguments, to be applied at the
            corresponding leaves of the pytrees with extra paths.
        tree (pytree): A pytree to be mapped over, with each leaf providing the first positional
            argument to function ``func``.
        rests (tuple of pytree): A tuple of pytrees, they should have a common suffix structure with
            each other and with ``tree``.
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
        A new pytree with the structure as the common suffix structure of ``tree`` and ``rests`` but
        with the value at each leaf given by ``func(p, x, *xs)`` where ``(p, x)`` are the path and
        value at the corresponding leaf (may be broadcasted) in and ``xs`` is the tuple of values at
        corresponding leaves (may be broadcasted) in ``rests``.
    """
    if not rests:
        return tree_map_with_path(func, tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    broadcasted_tree = tree
    broadcasted_rests = list(rests)
    for _ in range(2):
        for i, rest in enumerate(rests):
            broadcasted_tree, broadcasted_rests[i] = tree_broadcast_common(broadcasted_tree, rest, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    return tree_map_with_path(func, broadcasted_tree, *broadcasted_rests, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)