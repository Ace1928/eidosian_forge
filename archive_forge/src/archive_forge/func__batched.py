import math
import operator
from collections import deque
from collections.abc import Sized
from functools import partial, reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def _batched(iterable, n, *, strict=False):
    """Batch data into tuples of length *n*. If the number of items in
    *iterable* is not divisible by *n*:
    * The last batch will be shorter if *strict* is ``False``.
    * :exc:`ValueError` will be raised if *strict* is ``True``.

    >>> list(batched('ABCDEFG', 3))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

    On Python 3.13 and above, this is an alias for :func:`itertools.batched`.
    """
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch