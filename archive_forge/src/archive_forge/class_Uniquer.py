import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class Uniquer(dict):

    def __init__(self):
        self.counter = _Counter()

    def __missing__(self, key):
        self[key] = count = self.counter()
        return count