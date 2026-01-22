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
def _to_keys(self, key):
    prefix = self.prefix
    if prefix:
        pkey = prefix + key if not key.startswith(prefix) else key
        return (match_case(pkey, prefix), key)
    return (key,)