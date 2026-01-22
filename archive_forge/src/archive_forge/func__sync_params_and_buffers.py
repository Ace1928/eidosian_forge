import dataclasses
import traceback
from typing import Any, Callable, Container, Dict, List, Optional, OrderedDict, Tuple, TypeVar, overload
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence
def _sync_params_and_buffers(process_group: dist.ProcessGroup, module_states: List[torch.Tensor], broadcast_bucket_size: int, src: int) -> None:
    """Synchronize ``module_states`` (list of tensors) across all processes by broadcasting them from rank 0."""
    if len(module_states) > 0:
        dist._broadcast_coalesced(process_group, module_states, broadcast_bucket_size, src)