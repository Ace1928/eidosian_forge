import math
import unittest
from cachetools import TLRUCache
from . import CacheTestMixin
class TLRUTestCache(TLRUCache):

    def default_ttu(_key, _value, _time):
        return math.inf

    def __init__(self, maxsize, ttu=default_ttu, **kwargs):
        TLRUCache.__init__(self, maxsize, ttu, timer=Timer(), **kwargs)