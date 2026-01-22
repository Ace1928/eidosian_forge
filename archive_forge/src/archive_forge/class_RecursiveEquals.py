import unittest
import cachetools.func
class RecursiveEquals:

    def __init__(self, use_cache):
        self._use_cache = use_cache

    def __hash__(self):
        return hash(self._use_cache)

    def __eq__(self, other):
        if self._use_cache:
            cached(self)
        return self._use_cache == other._use_cache