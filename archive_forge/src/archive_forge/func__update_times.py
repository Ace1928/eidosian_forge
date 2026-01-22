import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def _update_times(timer, times):
    if times is None:
        return
    for dest, start, end in (('fragment', 'start fmcs', 'end fragment'), ('select', 'end fragment', 'end select'), ('enumerate', 'end select', 'end fmcs'), ('best_found', 'start fmcs', 'new best'), ('mcs', 'start fmcs', 'end fmcs')):
        try:
            diff = timer.mark_times[end] - timer.mark_times[start]
        except KeyError:
            diff = None
        times[dest] = diff