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
def treespec_entry(treespec: PyTreeSpec, index: int) -> Any:
    """Return the entry of a treespec at the given index.

    See also :func:`treespec_entries`, :func:`treespec_children`, and :meth:`PyTreeSpec.entry`.
    """
    return treespec.entry(index)