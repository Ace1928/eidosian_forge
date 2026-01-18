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
def treespec_ordereddict(mapping: Mapping[Any, PyTreeSpec] | Iterable[tuple[Any, PyTreeSpec]]=(), *, none_is_leaf: bool=False, namespace: str='', **kwargs: PyTreeSpec) -> PyTreeSpec:
    """Make an OrderedDict treespec from an OrderedDict of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> treespec_ordereddict({'a': treespec_leaf(), 'b': treespec_leaf()})
    PyTreeSpec(OrderedDict([('a', *), ('b', *)]))
    >>> treespec_ordereddict([('b', treespec_leaf()), ('c', treespec_leaf()), ('a', treespec_none())])
    PyTreeSpec(OrderedDict([('b', *), ('c', *), ('a', None)]))
    >>> treespec_ordereddict()
    PyTreeSpec(OrderedDict([]))
    >>> treespec_ordereddict(a=treespec_leaf(), b=treespec_tuple([treespec_leaf(), treespec_leaf()]))
    PyTreeSpec(OrderedDict([('a', *), ('b', (*, *))]))
    >>> treespec_ordereddict({'a': treespec_leaf(), 'b': tree_structure([1, 2])})
    PyTreeSpec(OrderedDict([('a', *), ('b', [*, *])]))
    >>> treespec_ordereddict({'a': treespec_leaf(), 'b': tree_structure([1, 2], none_is_leaf=True)})
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        mapping (mapping of PyTreeSpec, optional): A mapping of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing an OrderedDict node with the given children.
    """
    return _C.make_from_collection(OrderedDict(mapping, **kwargs), none_is_leaf, namespace)