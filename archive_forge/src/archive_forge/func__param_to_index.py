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
def _param_to_index(self) -> Dict[torch.Tensor, int]:
    """
        :class:`dict` mapping parameters to their indices in the global optimizer state.

        NOTE: This assumes that the global optimizer state's indexing (in
        ``state_dict``) follows a linear ordering over the parameter groups.
        """
    if len(self._param_to_index_cache) == 0:
        self._param_to_index_cache = {p: i for i, p in enumerate(chain(*(g['params'] for g in self.param_groups)))}
    return self._param_to_index_cache