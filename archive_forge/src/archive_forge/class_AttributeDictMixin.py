import time
from collections import OrderedDict as _OrderedDict
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping, MutableSet, Sequence
from heapq import heapify, heappop, heappush
from itertools import chain, count
from queue import Empty
from typing import Any, Dict, Iterable, List  # noqa
from .functional import first, uniq
from .text import match_case
class AttributeDictMixin:
    """Mixin for Mapping interface that adds attribute access.

    I.e., `d.key -> d[key]`).
    """

    def __getattr__(self, k):
        """`d.key -> d[key]`."""
        try:
            return self[k]
        except KeyError:
            raise AttributeError(f'{type(self).__name__!r} object has no attribute {k!r}')

    def __setattr__(self, key: str, value) -> None:
        """`d[key] = value -> d.key = value`."""
        self[key] = value