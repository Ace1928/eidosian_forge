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
def user_autotune(configs, triton_meta, filename=None, inductor_meta=None):
    """
    Compile a user defined triton kernel
    """
    defaults = inspect.signature(triton.Config).parameters
    default_num_stages = defaults['num_stages'].default
    default_num_warps = defaults['num_warps'].default
    if len(configs) == 0:
        configs = [triton.Config({}, num_stages=default_num_stages, num_warps=default_num_warps)]
    else:
        configs = [triton.Config(c.get('kwargs', {}), num_stages=c.get('num_stages', default_num_stages), num_warps=c.get('num_warps', default_num_warps)) for c in configs]
    return cached_autotune(None, configs, triton_meta=triton_meta, heuristic_type=HeuristicType.USER_AUTOTUNE, filename=filename, inductor_meta=inductor_meta)