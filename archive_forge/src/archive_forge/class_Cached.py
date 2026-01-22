import operator
import unittest
from cachetools import LRUCache, cachedmethod, keys
class Cached:

    def __init__(self, cache, count=0):
        self.cache = cache
        self.count = count

    @cachedmethod(operator.attrgetter('cache'))
    def get(self, value):
        count = self.count
        self.count += 1
        return count

    @cachedmethod(operator.attrgetter('cache'), key=keys.typedkey)
    def get_typed(self, value):
        count = self.count
        self.count += 1
        return count