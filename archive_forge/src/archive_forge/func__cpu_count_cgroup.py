import os
import sys
import math
import subprocess
import traceback
import warnings
import multiprocessing as mp
from multiprocessing import get_context as mp_get_context
from multiprocessing.context import BaseContext
from .process import LokyProcess, LokyInitMainProcess
def _cpu_count_cgroup(os_cpu_count):
    cpu_max_fname = '/sys/fs/cgroup/cpu.max'
    cfs_quota_fname = '/sys/fs/cgroup/cpu/cpu.cfs_quota_us'
    cfs_period_fname = '/sys/fs/cgroup/cpu/cpu.cfs_period_us'
    if os.path.exists(cpu_max_fname):
        with open(cpu_max_fname) as fh:
            cpu_quota_us, cpu_period_us = fh.read().strip().split()
    elif os.path.exists(cfs_quota_fname) and os.path.exists(cfs_period_fname):
        with open(cfs_quota_fname) as fh:
            cpu_quota_us = fh.read().strip()
        with open(cfs_period_fname) as fh:
            cpu_period_us = fh.read().strip()
    else:
        cpu_quota_us = 'max'
        cpu_period_us = 100000
    if cpu_quota_us == 'max':
        return os_cpu_count
    else:
        cpu_quota_us = int(cpu_quota_us)
        cpu_period_us = int(cpu_period_us)
        if cpu_quota_us > 0 and cpu_period_us > 0:
            return math.ceil(cpu_quota_us / cpu_period_us)
        else:
            return os_cpu_count