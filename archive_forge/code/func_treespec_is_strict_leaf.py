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
def treespec_is_strict_leaf(treespec: PyTreeSpec) -> bool:
    """Return whether the treespec is a strict leaf.

    See also :func:`treespec_is_leaf` and :meth:`PyTreeSpec.is_leaf`.

    This function respects the ``none_is_leaf`` setting in the treespec. It is equivalent to
    ``treespec.is_leaf(strict=True)``. It will return :data:`True` if and only if the treespec
    represents a strict leaf.

    >>> treespec_is_strict_leaf(tree_structure(1))
    True
    >>> treespec_is_strict_leaf(tree_structure((1, 2)))
    False
    >>> treespec_is_strict_leaf(tree_structure(None))
    False
    >>> treespec_is_strict_leaf(tree_structure(None, none_is_leaf=False))
    False
    >>> treespec_is_strict_leaf(tree_structure(None, none_is_leaf=True))
    True
    >>> treespec_is_strict_leaf(tree_structure(()))
    False
    >>> treespec_is_strict_leaf(tree_structure([]))
    False

    Args:
        treespec (PyTreeSpec): A treespec.

    Returns:
        :data:`True` if the treespec represents a strict leaf, otherwise, :data:`False`.
    """
    return treespec.num_nodes == 1 and treespec.num_leaves == 1