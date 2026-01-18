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

        Coordinate descent tuning can be run with or without max-autotune.

        The only difference between these two is the starting config for coordinate_descent tuning.
        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4
        and max-autotune figure out C3 is the best.

        Then if coordinate descnt tuning is run with max-autotune disabled, it will start from C1;
        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.
        