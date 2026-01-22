from __future__ import annotations
import collections
import dataclasses
import itertools
import pprint
from typing import Any, Dict, Iterable, List, Optional, Protocol
import sympy
import torch
from .. import config, ir
from ..utils import cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
@dataclasses.dataclass
class TemporalSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains a list of allocations not overlapping in LiveRanges.

    Invariant: no pair (a,b) in self.allocations will have:
         a.get_live_ranges().overlaps(b.get_live_ranges())
    """
    allocations: List[AllocationTreeNode]

    def _allocate(self, block: Allocation, is_last: bool):
        slot_size = self.get_size_hint()
        block_size = block.get_size_hint()
        if not is_last and block_size > slot_size:
            return False
        block_live = block.get_live_ranges()
        overlapping = [s for s in self.allocations if s.get_live_ranges().overlaps(block_live)]
        if len(overlapping) > 1:
            return False
        elif len(overlapping) == 1:
            return overlapping[0].allocate(block, is_last)
        else:
            block.mark_allocated()
            if len(self.allocations) == 1 and isinstance(self.allocations[-1], Empty):
                self.allocations.pop()
            if slot_size == block_size:
                self.allocations.append(block)
            elif slot_size > block_size:
                self.allocations.append(SpatialSplit.create(block, slot_size - block_size))
            else:
                assert is_last
                self.allocations = [*(SpatialSplit.create(a, block_size - slot_size) for a in self.allocations), block]
            return True

    @cache_on_self
    def get_live_ranges(self) -> LiveRanges:
        return LiveRanges(itertools.chain.from_iterable((x.get_live_ranges().ranges for x in self.allocations)))

    @cache_on_self
    def get_size_hint(self) -> int:
        if not self.allocations:
            return 0
        return max((x.get_size_hint() for x in self.allocations))

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        if not self.allocations:
            return 0
        return sympy.Max(*[x.get_symbolic_size() for x in self.allocations])

    def is_empty(self):
        return len(self.allocations) == 1 and self.allocations[0].is_empty()

    def finalize(self, pool, offset):
        self.allocations = [block.finalize(pool, offset) for block in self.allocations]
        self.clear_cache()
        if len(self.allocations) == 1:
            return self.allocations[0]
        return self