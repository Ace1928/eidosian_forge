import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
def __update(self, key):
    try:
        self.__order.move_to_end(key, last=False)
    except KeyError:
        self.__order[key] = None