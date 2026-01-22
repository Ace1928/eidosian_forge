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
@property
def _index_to_param(self) -> List[torch.Tensor]:
    """List mapping parameter indices in the global optimizer scheme to the actual params."""
    if len(self._index_to_param_cache) == 0:
        self._index_to_param_cache = list(chain(*(g['params'] for g in self.param_groups)))
    return self._index_to_param_cache