import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def _make_execution_plan(blocking: _Blocking, location_map_a: _BlockLocationMap, location_map_b: _BlockLocationMap) -> _ExecutionPlan:
    i_partitions = blocking.i_partitions
    j_partitions = blocking.j_partitions
    k_partitions = blocking.k_partitions
    plan: _ExecutionPlan = []
    for i_range in zip(i_partitions, i_partitions[1:]):
        for j_range in zip(j_partitions, j_partitions[1:]):
            for k_range in zip(k_partitions, k_partitions[1:]):
                block_a = (i_range + (1,), k_range + (1,))
                block_b = (k_range + (1,), j_range + (1,))
                devices_a = set(location_map_a[block_a].keys())
                devices_b = set(location_map_b[block_b].keys())
                intersection = devices_a & devices_b
                if intersection:
                    dev = intersection.pop()
                    plan.append((block_a, block_b, dev))
                else:
                    raise RuntimeError(f'There is no device that can perform multiplication between block {block_a} and {block_b}')
    return plan