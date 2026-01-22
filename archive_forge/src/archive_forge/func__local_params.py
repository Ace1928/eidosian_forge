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
def _local_params(self) -> List[torch.Tensor]:
    """Iterable which goes through the parameters that this rank owns"""
    if self.__local_params is None:
        self.__local_params = list(chain(*[list(filter(lambda x: x.grad is not None, device_params[self.rank])) for device_params in self._per_device_params.values()]))
    return self.__local_params