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
def treespec_child(treespec: PyTreeSpec, index: int) -> PyTreeSpec:
    """Return the treespec of the child of a treespec at the given index.

    See also :func:`treespec_children`, :func:`treespec_entries`, and :meth:`PyTreeSpec.child`.
    """
    return treespec.child(index)