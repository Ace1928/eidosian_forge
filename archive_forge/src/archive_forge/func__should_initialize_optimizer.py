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
def _should_initialize_optimizer(self) -> bool:
    """
        Used to indicate whether we should initialize an optimizer.

        When this is off, we can save memory and use larger batches.
        """
    if self.opt.get('interactive_mode'):
        return False
    datatype = self.opt.get('datatype', '')
    is_train = 'train' in datatype and 'evalmode' not in datatype
    return is_train