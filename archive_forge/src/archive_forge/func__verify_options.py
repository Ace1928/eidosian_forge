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
def _verify_options(model: nn.Module, optims: Tuple[torch.optim.Optimizer, ...], optim_only: bool, *, submodules: Optional[Set[nn.Module]]=None, options: Optional[StateDictOptions]=None) -> _StateDictInfo:
    """
    Verify the model and options passed by the user and generates _StateDictInfo.
    """
    if optim_only and (not optims):
        raise RuntimeError('Optimizers are not passed in but optim_only is set to True.')
    options = options or StateDictOptions()
    fqn_param_mapping: Dict[Union[str, torch.Tensor], Union[Set[str], torch.Tensor]] = {}
    all_fqns = set()
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        fqns = _get_fqns(model, name)
        fqn_param_mapping[param] = fqns
        for fqn in fqns:
            fqn_param_mapping[fqn] = param
            all_fqns.add(fqn)
    submodule_prefixes = set()
    if submodules:
        submodules = set(submodules)
        for name, module in model.named_modules():
            if module not in submodules:
                continue
            fqns = _get_fqns(model, name)
            assert len(fqns) == 1, 'Submodule FQN should only have 1 instance'
            for fqn in fqns:
                submodule_prefixes.add(f'{fqn}.')
    fsdp_modules = FSDP.fsdp_modules(model)
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
    fsdp_context: Callable
    if fsdp_modules:
        if options.full_state_dict:
            state_dict_config = FullStateDictConfig(offload_to_cpu=options.cpu_offload, rank0_only=options.cpu_offload)
            optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=options.cpu_offload, rank0_only=options.cpu_offload)
            state_dict_type = StateDictType.FULL_STATE_DICT
        else:
            state_dict_config = ShardedStateDictConfig()
            optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=options.cpu_offload)
            state_dict_type = StateDictType.SHARDED_STATE_DICT
        fsdp_context = functools.partial(FSDP.state_dict_type, module=model, state_dict_type=state_dict_type, state_dict_config=state_dict_config, optim_state_dict_config=optim_state_dict_config)
    else:
        fsdp_context = contextlib.nullcontext
    return _StateDictInfo(**asdict(options), fqn_param_mapping=fqn_param_mapping, all_fqns=all_fqns, submodule_prefixes=submodule_prefixes, fsdp_context=fsdp_context, fsdp_modules=cast(List[nn.Module], fsdp_modules), handle_model=not optim_only, handle_optim=len(optims) > 0)