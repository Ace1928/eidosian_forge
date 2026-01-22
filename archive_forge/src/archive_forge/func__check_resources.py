import os
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, pool
from traceback import format_exception
import sys
from logging import INFO
import gc
from copy import deepcopy
import numpy as np
from ... import logging
from ...utils.profiler import get_system_total_memory_gb
from ..engine import MapNode
from .base import DistributedPluginBase
def _check_resources(self, running_tasks):
    """
        Make sure there are resources available
        """
    free_memory_gb = self.memory_gb
    free_processors = self.processors
    for _, jobid in running_tasks:
        free_memory_gb -= min(self.procs[jobid].mem_gb, free_memory_gb)
        free_processors -= min(self.procs[jobid].n_procs, free_processors)
    return (free_memory_gb, free_processors)