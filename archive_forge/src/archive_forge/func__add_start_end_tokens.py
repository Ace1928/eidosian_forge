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
def _add_start_end_tokens(self, vec, add_start=False, add_end=False):
    """
        Add start and end tokens to a list or tensor.
        """
    if isinstance(vec, torch.Tensor):
        if len(vec.shape) != 1:
            raise Exception('_add_start_end_tokens expects a 1D tensor')
        tensors = [vec]
        if add_start:
            tensors.insert(0, vec.new_tensor([self.START_IDX]))
        if add_end:
            tensors.append(vec.new_tensor([self.END_IDX]))
        return torch.cat(tensors, 0)
    if add_start:
        vec.insert(0, self.START_IDX)
    if add_end:
        vec.append(self.END_IDX)
    return vec