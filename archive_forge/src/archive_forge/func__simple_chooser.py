import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _simple_chooser(queue, remaining):
    """Default contraction chooser that simply takes the minimum cost option.
    """
    cost, k1, k2, k12 = heapq.heappop(queue)
    if k1 not in remaining or k2 not in remaining:
        return None
    return (cost, k1, k2, k12)