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
def _get_is_trainable_mask(self) -> List[bool]:
    """Return a boolean mask indicating if each parameter is trainable (``requires_grad``) or not."""
    return list(map(_is_trainable, self._all_params))