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
def treespec_from_collection(collection: CustomTreeNode[PyTreeSpec], *, none_is_leaf: bool=False, namespace: str='') -> PyTreeSpec:
    """Make a treespec from a collection of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_from_collection(None)
    PyTreeSpec(None)
    >>> treespec_from_collection(None, none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)
    >>> treespec_from_collection(object())
    PyTreeSpec(*)
    >>> treespec_from_collection([treespec_leaf(), treespec_none()])
    PyTreeSpec([*, None])
    >>> treespec_from_collection({'a': treespec_leaf(), 'b': treespec_none()})
    PyTreeSpec({'a': *, 'b': None})
    >>> treespec_from_collection(deque([treespec_leaf(), tree_structure({'a': 1, 'b': 2})], maxlen=5))
    PyTreeSpec(deque([*, {'a': *, 'b': *}], maxlen=5))
    >>> treespec_from_collection({'a': treespec_leaf(), 'b': (treespec_leaf(), treespec_none())})
    Traceback (most recent call last):
        ...
    ValueError: Expected a(n) dict of PyTreeSpec(s), got {'a': PyTreeSpec(*), 'b': (PyTreeSpec(*), PyTreeSpec(None))}.
    >>> treespec_from_collection([treespec_leaf(), tree_structure({'a': 1, 'b': 2}, none_is_leaf=True)])
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.


    Args:
        collection (collection of PyTreeSpec): A collection of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing the same structure of the collection with the given children.
    """
    return _C.make_from_collection(collection, none_is_leaf, namespace)