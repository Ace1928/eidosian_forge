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
def zip_broadcast(*objects, scalar_types=(str, bytes), strict=False):
    """A version of :func:`zip` that "broadcasts" any scalar
    (i.e., non-iterable) items into output tuples.

    >>> iterable_1 = [1, 2, 3]
    >>> iterable_2 = ['a', 'b', 'c']
    >>> scalar = '_'
    >>> list(zip_broadcast(iterable_1, iterable_2, scalar))
    [(1, 'a', '_'), (2, 'b', '_'), (3, 'c', '_')]

    The *scalar_types* keyword argument determines what types are considered
    scalar. It is set to ``(str, bytes)`` by default. Set it to ``None`` to
    treat strings and byte strings as iterable:

    >>> list(zip_broadcast('abc', 0, 'xyz', scalar_types=None))
    [('a', 0, 'x'), ('b', 0, 'y'), ('c', 0, 'z')]

    If the *strict* keyword argument is ``True``, then
    ``UnequalIterablesError`` will be raised if any of the iterables have
    different lengths.
    """

    def is_scalar(obj):
        if scalar_types and isinstance(obj, scalar_types):
            return True
        try:
            iter(obj)
        except TypeError:
            return True
        else:
            return False
    size = len(objects)
    if not size:
        return
    iterables, iterable_positions = ([], [])
    scalars, scalar_positions = ([], [])
    for i, obj in enumerate(objects):
        if is_scalar(obj):
            scalars.append(obj)
            scalar_positions.append(i)
        else:
            iterables.append(iter(obj))
            iterable_positions.append(i)
    if len(scalars) == size:
        yield tuple(objects)
        return
    zipper = _zip_equal if strict else zip
    for item in zipper(*iterables):
        new_item = [None] * size
        for i, elem in zip(iterable_positions, item):
            new_item[i] = elem
        for i, elem in zip(scalar_positions, scalars):
            new_item[i] = elem
        yield tuple(new_item)