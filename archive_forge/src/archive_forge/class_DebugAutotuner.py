import builtins
import copy
import functools
import hashlib
import inspect
import json
import logging
import math
import operator
import os
import os.path
import re
import threading
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import torch
import torch.autograd.profiler as autograd_profiler
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import dynamo_timed
from torch.utils._triton import has_triton, has_triton_package
from . import config
from .codecache import cache_dir, CudaKernelParamCache
from .coordinate_descent_tuner import CoordescTuner
from .ir import ReductionHint, TileHint
from .utils import (
class DebugAutotuner(CachingAutotuner):

    def __init__(self, *args, regex_filter='', **kwargs):
        self.regex_filter = regex_filter
        super().__init__(*args, **kwargs)
        self.cached = None

    def run(self, *args, grid, stream):
        possible_names = _find_names(self)
        kernel_name = f'{max(possible_names, key=len)}'
        if not re.match(self.regex_filter, kernel_name):
            return
        super().run(*args, grid=grid, stream=stream)
        launcher, = self.launchers
        if self.cached is None:
            ms = self.bench(launcher, *args, grid=grid)
            num_in_out_ptrs = len([arg_name for arg_name in self.fn.arg_names if arg_name.startswith('in_out_ptr')])
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1000000000.0
            gb_per_s = num_gb / (ms / 1000.0)
            self.cached = (ms, num_gb, gb_per_s, kernel_name)
        else:
            ms, num_gb, gb_per_s, kernel_name = self.cached
        collected_calls.append((ms, num_gb, gb_per_s, kernel_name))
        print(create_bandwidth_info_str(ms, num_gb, gb_per_s, suffix=f' \t {kernel_name}'))