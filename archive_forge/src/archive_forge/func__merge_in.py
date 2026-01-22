import collections
from collections import abc
import itertools
def _merge_in(target, iterable=None, sentinel=_sentinel):
    """Merges iterable into the target and returns the target."""
    if iterable is not None:
        for value in iterable:
            target.setdefault(value, sentinel)
    return target