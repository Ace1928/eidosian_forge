import contextlib
import functools
import gc
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._state_dict_utils import (
from torch.distributed.fsdp import (
from torch.distributed.fsdp._common_utils import (
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel as DDP
def _load_model_state_dict(model: nn.Module, state_dict: Dict[str, ValueType], info: _StateDictInfo) -> _IncompatibleKeys:
    if not info.handle_model or not state_dict:
        return _IncompatibleKeys({}, {})
    for key, _ in chain(model.named_parameters(), model.named_buffers()):
        fqns = _get_fqns(model, key)
        fqns_with_ddp_prefix = _get_fqns(model, key, skip_ddp_prefix=False)
        for fqn, fqn_with_ddp_prefix in zip(fqns, fqns_with_ddp_prefix):
            if fqn != fqn_with_ddp_prefix:
                state_dict[fqn_with_ddp_prefix] = state_dict.pop(fqn)
    with info.fsdp_context():
        return cast(_IncompatibleKeys, _state_dict_fn(model, 'load_state_dict')(state_dict=state_dict, strict=info.strict))