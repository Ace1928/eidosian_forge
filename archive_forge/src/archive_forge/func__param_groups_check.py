import logging
import warnings
from copy import deepcopy
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Union, overload
import torch
import torch.nn as nn
from torch import optim
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
def _param_groups_check(self):
    if self.param_groups is not None:
        for param_group in self.param_groups:
            assert isinstance(param_group, dict), 'param group must be a dict'
            assert 'params' in param_group, 'param group must contain key params'
            params = param_group['params']
            if isinstance(params, torch.Tensor):
                params = [params]
            params = list(params)
            for param in params:
                if not isinstance(param, torch.Tensor):
                    raise TypeError('optimizer can only optimize Tensors, but one of the params is ' + torch.typename(param))
            param_group['params'] = params