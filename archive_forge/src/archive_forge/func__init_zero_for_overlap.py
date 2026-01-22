import collections
import copy
import enum
import inspect
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer
def _init_zero_for_overlap(self) -> None:
    """Perform a delayed initialization of the local optimizer and the supporting data structures."""
    assert self._overlap_with_ddp, '`_init_zero_for_overlap()` should only be called when `overlap_with_ddp=True`'
    self._overlap_info.status = _OverlapStatus.INITIALIZED
    self._clear_cache()
    self._partition_parameters(self._overlap_info.params_per_rank)
    self._build_ddp_param_buckets()
    self._init_local_optimizer()