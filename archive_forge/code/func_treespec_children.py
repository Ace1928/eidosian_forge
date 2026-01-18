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
def treespec_children(treespec: PyTreeSpec) -> list[PyTreeSpec]:
    """Return a list of treespecs for the children of a treespec.

    See also :func:`treespec_child`, :func:`treespec_paths`, :func:`treespec_entries`,
    and :meth:`PyTreeSpec.children`.
    """
    return treespec.children()