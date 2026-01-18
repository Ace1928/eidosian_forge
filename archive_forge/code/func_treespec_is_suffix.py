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
def treespec_is_suffix(treespec: PyTreeSpec, other_treespec: PyTreeSpec, strict: bool=False) -> bool:
    """Return whether ``treespec`` is a suffix of ``other_treespec``.

    See also :func:`treespec_is_suffix` :meth:`PyTreeSpec.is_suffix`.
    """
    return treespec.is_suffix(other_treespec, strict=strict)