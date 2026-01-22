import atexit
import collections
import contextlib
import copy
import cProfile
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import inspect
import itertools
import linecache
import logging
import math
import operator
import os
import pstats
import subprocess
import sys
import textwrap
import threading
import time
import types
import typing
import weakref
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
import importlib
import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map_only
from torch._subclasses import (  # noqa: F401
@dataclasses.dataclass
class CompilationMetrics:
    frame_key: str
    co_name: str
    co_filename: str
    co_firstlineno: int
    cache_size: int
    accumulated_cache_size: int
    guard_count: Optional[int]
    graph_op_count: Optional[int]
    graph_node_count: Optional[int]
    graph_input_count: Optional[int]
    entire_frame_compile_time_s: Optional[float]
    backend_compile_time_s: Optional[float]
    fail_reason: Optional[str]
    non_compliant_ops: Set[str]
    compliant_custom_ops: Set[str]