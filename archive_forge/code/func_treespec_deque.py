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
def treespec_deque(iterable: Iterable[PyTreeSpec]=(), maxlen: int | None=None, *, none_is_leaf: bool=False, namespace: str='') -> PyTreeSpec:
    """Make a deque treespec from a deque of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_deque([treespec_leaf(), treespec_leaf()])
    PyTreeSpec(deque([*, *]))
    >>> treespec_deque([treespec_leaf(), treespec_leaf(), treespec_none()], maxlen=5)
    PyTreeSpec(deque([*, *, None], maxlen=5))
    >>> treespec_deque()
    PyTreeSpec(deque([]))
    >>> treespec_deque([treespec_leaf(), treespec_tuple([treespec_leaf(), treespec_leaf()])])
    PyTreeSpec(deque([*, (*, *)]))
    >>> treespec_deque([treespec_leaf(), tree_structure({'a': 1, 'b': 2})], maxlen=5)
    PyTreeSpec(deque([*, {'a': *, 'b': *}], maxlen=5))
    >>> treespec_deque([treespec_leaf(), tree_structure({'a': 1, 'b': 2}, none_is_leaf=True)], maxlen=5)
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        iterable (iterable of PyTreeSpec, optional): A iterable of child treespecs. They must have
            the same ``node_is_leaf`` and ``namespace`` values.
        maxlen (int or None, optional): The maximum size of a deque or :data:`None` if unbounded.
            (default: :data:`None`)
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a deque node with the given children.
    """
    return _C.make_from_collection(deque(iterable, maxlen=maxlen), none_is_leaf, namespace)