from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
class SchedulerImportCheck(SchedulerPlugin):
    """Plugin to help record which modules are imported on the scheduler"""
    name = 'import-check'

    def __init__(self, pattern):
        self.pattern = pattern

    async def start(self, scheduler):
        self.start_modules = set()
        for mod in set(sys.modules):
            if not mod.startswith(self.pattern):
                self.start_modules.add(mod)
            else:
                sys.modules.pop(mod)