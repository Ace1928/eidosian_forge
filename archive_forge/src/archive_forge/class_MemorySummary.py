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
class MemorySummary(NamedTuple):
    """
    `MemorySummary` namedtuple otherwise with the fields:

        - `sequential`: a list of `MemoryState` namedtuple (see below) computed from the provided `memory_trace` by
          subtracting the memory after executing each line from the memory before executing said line.
        - `cumulative`: a list of `MemoryState` namedtuple (see below) with cumulative increase in memory for each line
          obtained by summing repeated memory increase for a line if it's executed several times. The list is sorted
          from the frame with the largest memory consumption to the frame with the smallest (can be negative if memory
          is released)
        - `total`: total memory increase during the full tracing as a `Memory` named tuple (see below). Line with
          memory release (negative consumption) are ignored if `ignore_released_memory` is `True` (default).
    """
    sequential: List[MemoryState]
    cumulative: List[MemoryState]
    current: List[MemoryState]
    total: Memory