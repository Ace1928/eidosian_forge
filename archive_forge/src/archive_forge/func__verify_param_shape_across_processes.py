import dataclasses
import traceback
from typing import Any, Callable, Container, Dict, List, Optional, OrderedDict, Tuple, TypeVar, overload
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence
def _verify_param_shape_across_processes(process_group: dist.ProcessGroup, tensors: List[torch.Tensor], logger: Optional[dist.Logger]=None):
    return dist._verify_params_across_processes(process_group, tensors, logger)