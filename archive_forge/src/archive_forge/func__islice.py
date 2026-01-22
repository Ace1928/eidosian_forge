import sys
import traceback
from bisect import bisect_left, bisect_right, insort
from itertools import chain, repeat, starmap
from math import log
from operator import add, eq, ne, gt, ge, lt, le, iadd
from textwrap import dedent
from functools import wraps
from sys import hexversion
def _islice(self, min_pos, min_idx, max_pos, max_idx, reverse):
    """Return an iterator that slices sorted list using two index pairs.

        The index pairs are (min_pos, min_idx) and (max_pos, max_idx), the
        first inclusive and the latter exclusive. See `_pos` for details on how
        an index is converted to an index pair.

        When `reverse` is `True`, values are yielded from the iterator in
        reverse order.

        """
    _lists = self._lists
    if min_pos > max_pos:
        return iter(())
    if min_pos == max_pos:
        if reverse:
            indices = reversed(range(min_idx, max_idx))
            return map(_lists[min_pos].__getitem__, indices)
        indices = range(min_idx, max_idx)
        return map(_lists[min_pos].__getitem__, indices)
    next_pos = min_pos + 1
    if next_pos == max_pos:
        if reverse:
            min_indices = range(min_idx, len(_lists[min_pos]))
            max_indices = range(max_idx)
            return chain(map(_lists[max_pos].__getitem__, reversed(max_indices)), map(_lists[min_pos].__getitem__, reversed(min_indices)))
        min_indices = range(min_idx, len(_lists[min_pos]))
        max_indices = range(max_idx)
        return chain(map(_lists[min_pos].__getitem__, min_indices), map(_lists[max_pos].__getitem__, max_indices))
    if reverse:
        min_indices = range(min_idx, len(_lists[min_pos]))
        sublist_indices = range(next_pos, max_pos)
        sublists = map(_lists.__getitem__, reversed(sublist_indices))
        max_indices = range(max_idx)
        return chain(map(_lists[max_pos].__getitem__, reversed(max_indices)), chain.from_iterable(map(reversed, sublists)), map(_lists[min_pos].__getitem__, reversed(min_indices)))
    min_indices = range(min_idx, len(_lists[min_pos]))
    sublist_indices = range(next_pos, max_pos)
    sublists = map(_lists.__getitem__, sublist_indices)
    max_indices = range(max_idx)
    return chain(map(_lists[min_pos].__getitem__, min_indices), chain.from_iterable(sublists), map(_lists[max_pos].__getitem__, max_indices))