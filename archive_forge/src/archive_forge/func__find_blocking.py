import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def _find_blocking(location_map_a: _BlockLocationMap, location_map_b: _BlockLocationMap) -> _Blocking:
    i_partitions: list[int] = []
    j_partitions: list[int] = []
    k_partitions: list[int] = []

    def add_to_partitions(indices, partitions):
        start, stop, step = indices
        if step != 1:
            raise RuntimeError('Step other than 1 is not supported')
        partitions.append(start)
        partitions.append(stop)
    for i_indices, k_indices in location_map_a.keys():
        add_to_partitions(i_indices, i_partitions)
        add_to_partitions(k_indices, k_partitions)
    for k_indices, j_indices in location_map_b.keys():
        add_to_partitions(k_indices, k_partitions)
        add_to_partitions(j_indices, j_partitions)

    def to_unique_sorted(partitions):
        if len(partitions) == 0:
            raise RuntimeError('Array has no chunk')
        partitions.sort()
        res = [partitions[0]]
        for x, y in zip(partitions, partitions[1:]):
            if x != y:
                res.append(y)
        return res
    i_partitions = to_unique_sorted(i_partitions)
    j_partitions = to_unique_sorted(j_partitions)
    k_partitions = to_unique_sorted(k_partitions)

    def check_indices(indices, partitions):
        start, stop, _ = indices
        if partitions.index(start) + 1 != partitions.index(stop):
            raise RuntimeError('Inconsistent index mapping')
    for i_indices, k_indices in location_map_a.keys():
        check_indices(i_indices, i_partitions)
        check_indices(k_indices, k_partitions)
    for k_indices, j_indices in location_map_b.keys():
        check_indices(k_indices, k_partitions)
        check_indices(j_indices, j_partitions)
    return _Blocking(i_partitions, j_partitions, k_partitions)