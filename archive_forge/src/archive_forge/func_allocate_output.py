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
def allocate_output(self, block: Allocation):
    """Outputs get different pools so memory gets freed properly"""
    pools = self.get_pools(block)
    if pools and config.memory_pool in ('outputs', 'combined'):
        pools[-1].allocate_at_end(block)
    else:
        block.mark_allocated()
        pools.append(AllocationPool(block.device, TemporalSplit([block]), can_expand=config.memory_pool == 'combined'))