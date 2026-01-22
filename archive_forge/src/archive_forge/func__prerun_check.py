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
def _prerun_check(self, graph):
    """Check if any node exceeds the available resources"""
    tasks_mem_gb = []
    tasks_num_th = []
    for node in graph.nodes():
        tasks_mem_gb.append(node.mem_gb)
        tasks_num_th.append(node.n_procs)
    if np.any(np.array(tasks_mem_gb) > self.memory_gb):
        logger.warning('Some nodes exceed the total amount of memory available (%0.2fGB).', self.memory_gb)
        if self.raise_insufficient:
            raise RuntimeError('Insufficient resources available for job')
    if np.any(np.array(tasks_num_th) > self.processors):
        logger.warning('Some nodes demand for more threads than available (%d).', self.processors)
        if self.raise_insufficient:
            raise RuntimeError('Insufficient resources available for job')