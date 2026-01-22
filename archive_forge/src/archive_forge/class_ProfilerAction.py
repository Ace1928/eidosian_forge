import gzip
import json
import os
import tempfile
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from warnings import warn
import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
class ProfilerAction(Enum):
    """
    Profiler actions that can be taken at the specified intervals
    """
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3