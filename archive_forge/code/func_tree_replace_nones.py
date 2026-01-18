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
def tree_replace_nones(sentinel: Any, tree: PyTree[T] | None, namespace: str='') -> PyTree[T]:
    """Replace :data:`None` in ``tree`` with ``sentinel``.

    See also :func:`tree_flatten` and :func:`tree_map`.

    >>> tree_replace_nones(0, {'a': 1, 'b': None, 'c': (2, None)})
    {'a': 1, 'b': 0, 'c': (2, 0)}
    >>> tree_replace_nones(0, None)
    0

    Args:
        sentinel (object): The value to replace :data:`None` with.
        tree (pytree): A pytree to be transformed.
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A new pytree with the same structure as ``tree`` but with :data:`None` replaced.
    """
    if tree is None:
        return sentinel
    return tree_map(lambda x: x if x is not None else sentinel, tree, none_is_leaf=True, namespace=namespace)