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
def treespec_paths(treespec: PyTreeSpec) -> list[tuple[Any, ...]]:
    """Return a list of paths to the leaves of a treespec.

    See also :func:`tree_flatten_with_path`, :func:`tree_paths`, and :meth:`PyTreeSpec.paths`.
    """
    return treespec.paths()