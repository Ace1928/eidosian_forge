import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _optimal_iterate(path, remaining, inputs, flops):
    if len(remaining) == 1:
        best['flops'] = flops
        best['ssa_path'] = path
        return
    for i, j in itertools.combinations(remaining, 2):
        if i > j:
            i, j = (j, i)
        key = (inputs[i], inputs[j])
        try:
            k12, flops12 = result_cache[key]
        except KeyError:
            k12, flops12 = result_cache[key] = calc_k12_flops(inputs, output, remaining, i, j, size_dict)
        new_flops = flops + flops12
        if new_flops >= best['flops']:
            continue
        if memory_limit not in _UNLIMITED_MEM:
            try:
                size12 = size_cache[k12]
            except KeyError:
                size12 = size_cache[k12] = helpers.compute_size_by_dict(k12, size_dict)
            if size12 > memory_limit:
                new_flops = flops + _compute_oversize_flops(inputs, remaining, output, size_dict)
                if new_flops < best['flops']:
                    best['flops'] = new_flops
                    best['ssa_path'] = path + (tuple(remaining),)
                continue
        _optimal_iterate(path=path + ((i, j),), inputs=inputs + (k12,), remaining=remaining - {i, j} | {len(inputs)}, flops=new_flops)