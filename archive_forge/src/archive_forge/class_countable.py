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
class countable:
    """Wrap *iterable* and keep a count of how many items have been consumed.

    The ``items_seen`` attribute starts at ``0`` and increments as the iterable
    is consumed:

        >>> iterable = map(str, range(10))
        >>> it = countable(iterable)
        >>> it.items_seen
        0
        >>> next(it), next(it)
        ('0', '1')
        >>> list(it)
        ['2', '3', '4', '5', '6', '7', '8', '9']
        >>> it.items_seen
        10
    """

    def __init__(self, iterable):
        self._it = iter(iterable)
        self.items_seen = 0

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._it)
        self.items_seen += 1
        return item