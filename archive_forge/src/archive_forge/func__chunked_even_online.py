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
def _chunked_even_online(iterable, n):
    buffer = []
    maxbuf = n + (n - 2) * (n - 1)
    for x in iterable:
        buffer.append(x)
        if len(buffer) == maxbuf:
            yield buffer[:n]
            buffer = buffer[n:]
    yield from _chunked_even_finite(buffer, len(buffer), n)