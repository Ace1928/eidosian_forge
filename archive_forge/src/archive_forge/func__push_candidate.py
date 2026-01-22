import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue, push_all, cost_fn):
    candidates = (_get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2, cost_fn) for k2 in k2s)
    if push_all:
        for candidate in candidates:
            heapq.heappush(queue, candidate)
    else:
        heapq.heappush(queue, min(candidates))