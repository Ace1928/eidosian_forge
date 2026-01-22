from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
class MemTimer(Process):
    """
    Fetch memory consumption from over a time interval
    """

    def __init__(self, monitor_pid, interval, pipe, backend, max_usage=False, *args, **kw):
        self.monitor_pid = monitor_pid
        self.interval = interval
        self.pipe = pipe
        self.cont = True
        self.backend = backend
        self.max_usage = max_usage
        self.n_measurements = 1
        self.timestamps = kw.pop('timestamps', False)
        self.include_children = kw.pop('include_children', False)
        self.mem_usage = [_get_memory(self.monitor_pid, self.backend, timestamps=self.timestamps, include_children=self.include_children)]
        super(MemTimer, self).__init__(*args, **kw)

    def run(self):
        self.pipe.send(0)
        stop = False
        while True:
            cur_mem = _get_memory(self.monitor_pid, self.backend, timestamps=self.timestamps, include_children=self.include_children)
            if not self.max_usage:
                self.mem_usage.append(cur_mem)
            else:
                self.mem_usage[0] = max(cur_mem, self.mem_usage[0])
            self.n_measurements += 1
            if stop:
                break
            stop = self.pipe.poll(self.interval)
        self.pipe.send(self.mem_usage)
        self.pipe.send(self.n_measurements)