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
@property
def _per_device_params(self) -> Dict[torch.device, List[List[Parameter]]]:
    """Sorted list of all the params, first per device then per rank.

        Within a list params are sorted per number of elements to allow for an easy bucketing.
        """
    if len(self.__per_device_params) == 0:
        for param_group in self.param_groups:
            for param in param_group['params']:
                device = param.device
                if self.__per_device_params.get(device) is None:
                    self.__per_device_params[device] = [[] for _ in range(self.world_size)]
                self.__per_device_params[device][self._param_to_rank[param]] += [param]
        for device in self.__per_device_params.keys():
            for rank_params in self.__per_device_params[device]:
                rank_params.sort(key=lambda x: x.numel())
    return self.__per_device_params