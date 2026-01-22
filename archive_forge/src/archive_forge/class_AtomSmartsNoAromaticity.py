import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
class AtomSmartsNoAromaticity(dict):

    def __missing__(self, eleno):
        value = _get_symbol(eleno)
        self[eleno] = value
        return value