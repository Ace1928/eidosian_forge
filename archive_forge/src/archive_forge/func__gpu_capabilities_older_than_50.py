from collections import OrderedDict
import copy
import io
from itertools import chain
import logging
from math import inf
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.autograd import profiler
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import SGD, Optimizer
from fairscale.internal.params import calc_grad_norm, get_global_rank, recursive_copy_to_device
from fairscale.nn.misc import ParamBucket
def _gpu_capabilities_older_than_50() -> bool:
    """Return True if the GPU's compute capability is older than SM50."""
    global _gpu_is_old
    if _gpu_is_old is None:
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(f'cuda:{i}')
            if major <= 5:
                _gpu_is_old = True
        if _gpu_is_old is None:
            _gpu_is_old = False
    return _gpu_is_old