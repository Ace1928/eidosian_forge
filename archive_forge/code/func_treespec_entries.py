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
def treespec_entries(treespec: PyTreeSpec) -> list[Any]:
    """Return a list of one-level entries of a treespec to its children.

    See also :func:`treespec_entry`, :func:`treespec_paths`, :func:`treespec_children`,
    and :meth:`PyTreeSpec.entries`.
    """
    return treespec.entries()