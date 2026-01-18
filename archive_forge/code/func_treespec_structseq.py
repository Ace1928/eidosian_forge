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
def treespec_structseq(structseq: PyStructSequence[PyTreeSpec], *, none_is_leaf: bool=False, namespace: str='') -> PyTreeSpec:
    """Make a PyStructSequence treespec from a PyStructSequence of child treespecs.

    See also :func:`tree_structure`, :func:`treespec_leaf`, and :func:`treespec_none`.

    Args:
        structseq (PyStructSequence of PyTreeSpec): A PyStructSequence of child treespecs. They must
            have the same ``node_is_leaf`` and ``namespace`` values.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec representing a PyStructSequence node with the given children.
    """
    if not is_structseq_instance(structseq):
        raise ValueError(f'Expected a PyStructSequence of PyTreeSpec(s), got {structseq!r}.')
    return _C.make_from_collection(structseq, none_is_leaf, namespace)