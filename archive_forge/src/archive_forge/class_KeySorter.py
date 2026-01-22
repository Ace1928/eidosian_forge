import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
class KeySorter:

    def __init__(self, obj, *args):
        self.obj = obj

    def __lt__(self, other):
        return self._keysort(self.obj, other.obj) < 0

    def _keysort(self, a, b):
        """Sorts two sets of sets, to order them ideally for
                    matching."""
        a = a.maxkeys
        b = b.maxkeys
        lendiffa = len(keys ^ a)
        lendiffb = len(keys ^ b)
        if lendiffa == 0 and lendiffb == 0:
            return 0
        if lendiffa == 0:
            return -1
        if lendiffb == 0:
            return 1
        if self._compare(lendiffa, lendiffb) != 0:
            return self._compare(lendiffa, lendiffb)
        if len(keys & b) == len(keys & a):
            return self._compare(len(a), len(b))
        else:
            return self._compare(len(keys & b), len(keys & a))

    def _compare(self, obj1, obj2):
        if obj1 < obj2:
            return -1
        elif obj1 < obj2:
            return 1
        else:
            return 0