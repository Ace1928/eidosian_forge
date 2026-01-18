import warnings
from collections import Counter, defaultdict, deque, abc
from collections.abc import Sequence
from functools import partial, reduce, wraps
from heapq import heapify, heapreplace, heappop
from itertools import (
from math import exp, factorial, floor, log
from queue import Empty, Queue
from random import random, randrange, uniform
from operator import itemgetter, mul, sub, gt, lt, ge, le
from sys import hexversion, maxsize
from time import monotonic
from .recipes import (
def unique_in_window(iterable, n, key=None):
    """Yield the items from *iterable* that haven't been seen recently.
    *n* is the size of the lookback window.

        >>> iterable = [0, 1, 0, 2, 3, 0]
        >>> n = 3
        >>> list(unique_in_window(iterable, n))
        [0, 1, 2, 3, 0]

    The *key* function, if provided, will be used to determine uniqueness:

        >>> list(unique_in_window('abAcda', 3, key=lambda x: x.lower()))
        ['a', 'b', 'c', 'd', 'a']

    The items in *iterable* must be hashable.

    """
    if n <= 0:
        raise ValueError('n must be greater than 0')
    window = deque(maxlen=n)
    uniques = set()
    use_key = key is not None
    for item in iterable:
        k = key(item) if use_key else item
        if k in uniques:
            continue
        if len(uniques) == n:
            uniques.discard(window[0])
        uniques.add(k)
        window.append(k)
        yield item