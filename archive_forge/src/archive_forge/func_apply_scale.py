import logging
from collections import abc, defaultdict
from typing import Any, Dict, Iterable, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup
def apply_scale(val: Union[torch.Tensor, Iterable[torch.Tensor]]):
    if isinstance(val, torch.Tensor):
        assert _is_supported_device(val)
        if len(stash) == 0:
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(val.device)
            assert self._scale is not None
            stash.append(_GeneralMultiDeviceReplicator(self._scale))
        scaled_val = val * stash[0].get(val.device)
        return scaled_val.type(val.dtype)
    if isinstance(val, abc.Iterable):
        iterator = map(apply_scale, val)
        if isinstance(val, (list, tuple)):
            return type(val)(iterator)
        return iterator
    raise ValueError('outputs must be a Tensor or an iterable of Tensors')