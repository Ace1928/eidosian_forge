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
def _sort_jobs(self, jobids, scheduler='tsort'):
    if scheduler == 'mem_thread':
        return sorted(jobids, key=lambda item: (self.procs[item].mem_gb, self.procs[item].n_procs))
    return jobids