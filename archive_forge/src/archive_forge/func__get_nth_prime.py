import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def _get_nth_prime(n):
    current_size = len(_primes)
    while current_size <= n:
        _primes.append(next(_prime_stream))
        current_size += 1
    return _primes[n]