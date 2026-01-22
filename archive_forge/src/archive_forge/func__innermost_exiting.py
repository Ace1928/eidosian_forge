import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
def _innermost_exiting(blk: RegionBlock) -> BasicBlock:
    while isinstance(blk, RegionBlock):
        blk = blk.subregion.graph[blk.exiting]
    return blk