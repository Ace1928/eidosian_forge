import collections
import functools
import math
import random
import time
from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import keys
class _UnboundCache(dict):

    @property
    def maxsize(self):
        return None

    @property
    def currsize(self):
        return len(self)