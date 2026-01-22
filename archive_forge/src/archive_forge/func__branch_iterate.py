import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _branch_iterate(path, inputs, remaining, flops, size):
    if len(remaining) == 1:
        self.best['size'] = size
        self.best['flops'] = flops
        self.best['ssa_path'] = path
        return

    def _assess_candidate(k1, k2, i, j):
        try:
            k12, flops12 = result_cache[k1, k2]
        except KeyError:
            k12, flops12 = result_cache[k1, k2] = calc_k12_flops(inputs, output, remaining, i, j, size_dict)
        try:
            size12 = size_cache[k12]
        except KeyError:
            size12 = size_cache[k12] = helpers.compute_size_by_dict(k12, size_dict)
        new_flops = flops + flops12
        new_size = max(size, size12)
        if not self.better(new_flops, new_size, self.best['flops'], self.best['size']):
            return None
        if new_flops < self.best_progress[len(inputs)]:
            self.best_progress[len(inputs)] = new_flops
        elif new_flops > self.cutoff_flops_factor * self.best_progress[len(inputs)]:
            return None
        if memory_limit not in _UNLIMITED_MEM and size12 > memory_limit:
            new_flops = flops + _compute_oversize_flops(inputs, remaining, output, size_dict)
            if new_flops < self.best['flops']:
                self.best['flops'] = new_flops
                self.best['ssa_path'] = path + (tuple(remaining),)
            return None
        size1, size2 = (size_cache[inputs[i]], size_cache[inputs[j]])
        cost = self.cost_fn(size12, size1, size2, k12, k1, k2)
        return (cost, flops12, new_flops, new_size, (i, j), k12)
    candidates = []
    for i, j in itertools.combinations(remaining, 2):
        if i > j:
            i, j = (j, i)
        k1, k2 = (inputs[i], inputs[j])
        if k1.isdisjoint(k2):
            continue
        candidate = _assess_candidate(k1, k2, i, j)
        if candidate:
            heapq.heappush(candidates, candidate)
    if not candidates:
        for i, j in itertools.combinations(remaining, 2):
            if i > j:
                i, j = (j, i)
            k1, k2 = (inputs[i], inputs[j])
            candidate = _assess_candidate(k1, k2, i, j)
            if candidate:
                heapq.heappush(candidates, candidate)
    bi = 0
    while (self.nbranch is None or bi < self.nbranch) and candidates:
        _, _, new_flops, new_size, (i, j), k12 = heapq.heappop(candidates)
        _branch_iterate(path=path + ((i, j),), inputs=inputs + (k12,), remaining=remaining - {i, j} | {len(inputs)}, flops=new_flops, size=new_size)
        bi += 1