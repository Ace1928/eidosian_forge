import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class starting_from(object):

    def __init__(self, left):
        self.left = left

    def __contains__(self, value):
        return self.left <= value