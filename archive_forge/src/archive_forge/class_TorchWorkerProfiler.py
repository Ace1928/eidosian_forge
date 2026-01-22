import collections
import logging
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from packaging.version import Version
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import (
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train._internal import session
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.session import get_accelerator, set_accelerator
from ray.util.annotations import Deprecated, PublicAPI
@Deprecated
class TorchWorkerProfiler:
    """Utility class for running PyTorch Profiler on a Train worker.

    Args:
        trace_dir (Optional[str]): The directory to store traces on the
           worker node. If ``None``, this will use a default temporary dir.
    """
    WORKER_TRACE_DIR_NAME = 'pytorch_profiler_worker_traces'

    def __init__(self, trace_dir: Optional[str]=None):
        raise DeprecationWarning('The `ray.train.torch.TorchWorkerProfiler` API is deprecated in Ray 2.0.')