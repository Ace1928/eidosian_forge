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
def value_chain(*args):
    """Yield all arguments passed to the function in the same order in which
    they were passed. If an argument itself is iterable then iterate over its
    values.

        >>> list(value_chain(1, 2, 3, [4, 5, 6]))
        [1, 2, 3, 4, 5, 6]

    Binary and text strings are not considered iterable and are emitted
    as-is:

        >>> list(value_chain('12', '34', ['56', '78']))
        ['12', '34', '56', '78']


    Multiple levels of nesting are not flattened.

    """
    for value in args:
        if isinstance(value, (str, bytes)):
            yield value
            continue
        try:
            yield from value
        except TypeError:
            yield value