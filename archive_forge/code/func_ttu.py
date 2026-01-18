import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
@property
def ttu(self):
    """The local time-to-use function used by the cache."""
    return self.__ttu