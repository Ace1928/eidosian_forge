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
def _control_local_metrics(self, enabled: bool=False, disabled: bool=False):
    """
        Used to temporarily disable local metrics.

        This is useful for things like when you need to call super(), but
        prevent the parent from recording some metric. For example, if you're
        forwarding a dummy batch or calling super() but still want to modify
        the output.

        You can compare this to torch.no_grad in its goal.
        """
    if not enabled ^ disabled:
        raise ValueError('You must provide exactly one of enabled or disabled to _control_local_metrics.')
    self.__local_metrics_enabled = enabled