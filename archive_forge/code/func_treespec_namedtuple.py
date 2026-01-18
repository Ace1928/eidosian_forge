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
def treespec_namedtuple(namedtuple: NamedTuple[PyTreeSpec], *, none_is_leaf: bool=False, namespace: str='') -> PyTreeSpec:
    """Make a namedtuple treespec from a namedtuple of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    >>> from collections import namedtuple
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=treespec_leaf()))
    PyTreeSpec(Point(x=*, y=*))
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=treespec_tuple([treespec_leaf(), treespec_leaf()])))
    PyTreeSpec(Point(x=*, y=(*, *)))
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=tree_structure([1, 2])))
    PyTreeSpec(Point(x=*, y=[*, *]))
    >>> treespec_namedtuple(Point(x=treespec_leaf(), y=tree_structure([1, 2], none_is_leaf=True)))
    Traceback (most recent call last):
        ...
    ValueError: Expected treespec(s) with `node_is_leaf=False`.

    Args:
        namedtuple (namedtuple of PyTreeSpec): A namedtuple of child treespecs. They must have the
            same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a dict node with the given children.
    """
    if not is_namedtuple_instance(namedtuple):
        raise ValueError(f'Expected a namedtuple of PyTreeSpec(s), got {namedtuple!r}.')
    return _C.make_from_collection(namedtuple, none_is_leaf, namespace)