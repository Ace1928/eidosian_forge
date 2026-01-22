from typing import Dict, Any, Union, List, Tuple, Optional
from abc import ABC, abstractmethod
import random
import os
import torch
import parlai.utils.logging as logging
from torch import optim
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.nn.lr_scheduler import ParlAILRScheduler
from parlai.core.message import Message
from parlai.utils.distributed import is_distributed
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.fp16 import (
from parlai.core.metrics import (
from parlai.utils.distributed import is_primary_worker
from parlai.utils.torch import argsort, compute_grad_norm, padded_tensor, atomic_save
def _gpu_usage(self):
    """
        Compute GPU memory usage.

        Includes both allocated and cached memory; this should be close to the
        output of nvidia-smi, but not reflect of how much is currently demanded
        by the program. It may be viewed as a rough approximation of
        worst-case-until-now.

        :return: Percent of allocated GPU memory as a fraction of available.
        """
    if not self.use_cuda:
        return None
    if self.opt['gpu'] == -1:
        devices = range(torch.cuda.device_count())
    else:
        devices = [self.opt['gpu']]
    memory_avail = 0
    memory_used = 0
    for dev in devices:
        props = torch.cuda.get_device_properties(dev)
        memory_avail += props.total_memory
        memory_used += torch.cuda.max_memory_allocated(dev)
        torch.cuda.reset_max_memory_allocated(dev)
    return memory_used / memory_avail