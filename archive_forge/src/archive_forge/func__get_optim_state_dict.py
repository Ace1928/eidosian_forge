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
def _get_optim_state_dict(model: nn.Module, optimizers: Tuple[torch.optim.Optimizer, ...], info: _StateDictInfo) -> OptimizerStateType:
    if not info.handle_optim:
        return {}
    optim_state_dict: OptimizerStateType = {STATE: {}, PG: []}
    for optim in optimizers:
        _init_optim_state(optim)
        osd = _state_dict_fn(optim, 'state_dict')()
        if info.fsdp_modules:
            with info.fsdp_context():
                osd = FSDP.optim_state_dict(model, optim, osd)
        else:
            params = list(chain.from_iterable((g[PARAMS] for g in optim.param_groups)))
            param_pid_mapping = dict(zip(params, range(len(params))))
            fqn_pid_mapping = {}
            for key, param in model.named_parameters():
                fqns = _get_fqns(model, key)
                assert len(fqns) == 1
                fqn = next(iter(fqns))
                if param not in param_pid_mapping:
                    continue
                pid = param_pid_mapping[param]
                fqn_pid_mapping[fqn] = pid
                fqn_pid_mapping[pid] = fqn
            for key in list(osd[STATE].keys()):
                fqn = fqn_pid_mapping[key]
                osd[STATE][fqn] = osd[STATE].pop(key)
            for group in osd[PG]:
                group[PARAMS] = [fqn_pid_mapping[pid] for pid in group[PARAMS]]
        if not osd:
            continue
        cast(DictValueType, optim_state_dict[STATE]).update(osd[STATE])
        cast(ListDictValueType, optim_state_dict[PG]).extend(osd[PG])
    if info.full_state_dict:
        ranks_only = tuple() if not info.cpu_offload else (0,)
        return _gather_state_dict(optim_state_dict, cpu_offload=info.cpu_offload, ranks_only=ranks_only)
    elif info.cpu_offload:
        return _offload_state_dict_to_cpu(optim_state_dict)
    else:
        return optim_state_dict