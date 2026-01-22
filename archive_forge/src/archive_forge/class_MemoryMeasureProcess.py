import copy
import csv
import linecache
import os
import platform
import sys
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from datetime import datetime
from multiprocessing import Pipe, Process, Queue
from multiprocessing.connection import Connection
from typing import Callable, Iterable, List, NamedTuple, Optional, Union
from .. import AutoConfig, PretrainedConfig
from .. import __version__ as version
from ..utils import is_psutil_available, is_py3nvml_available, is_tf_available, is_torch_available, logging
from .benchmark_args_utils import BenchmarkArguments
class MemoryMeasureProcess(Process):
    """
            `MemoryMeasureProcess` inherits from `Process` and overwrites its `run()` method. Used to measure the
            memory usage of a process
            """

    def __init__(self, process_id: int, child_connection: Connection, interval: float):
        super().__init__()
        self.process_id = process_id
        self.interval = interval
        self.connection = child_connection
        self.num_measurements = 1
        self.mem_usage = get_cpu_memory(self.process_id)

    def run(self):
        self.connection.send(0)
        stop = False
        while True:
            self.mem_usage = max(self.mem_usage, get_cpu_memory(self.process_id))
            self.num_measurements += 1
            if stop:
                break
            stop = self.connection.poll(self.interval)
        self.connection.send(self.mem_usage)
        self.connection.send(self.num_measurements)