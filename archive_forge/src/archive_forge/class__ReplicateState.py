import weakref
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.nn.parallel import DistributedDataParallel
from .contract import _get_registry, contract
class _ReplicateState(_State):

    def __init__(self) -> None:
        super().__init__()
        self.module: nn.Module = nn.ParameterList()
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()
        self._param_names: List[str] = []

    def _collect_params(self, module: nn.Module, ignored_modules: Set[nn.Module], ignored_params: Set[nn.Parameter], prefix: str=_ROOT_MODULE_PREFIX) -> None:
        if _is_fully_sharded(module):
            return
        if module in ignored_modules:
            return
        recurse_prefix = f'{prefix}.' if prefix != _ROOT_MODULE_PREFIX else _ROOT_MODULE_PREFIX
        for n, p in module.named_parameters(recurse=False):
            if p not in ignored_params:
                self._param_list.append(p)
                self._param_names.append(f'{recurse_prefix}{n}')
        for name, child_module in module.named_children():
            self._collect_params(child_module, ignored_modules, ignored_params, prefix=f'{recurse_prefix}{name}')

    def init(self, module: nn.Module, ignored_modules: Set[nn.Module], **kwargs) -> None:
        if _is_fully_sharded(module):
            raise RuntimeError('Cannot apply `replicate()` on a Module already managed by `fully_shard`')
        if self.has_initialized:
            return
        self.has_initialized = True
        self.module = module
        ignored_params = {p for m in ignored_modules for p in m.parameters()}
        self._collect_params(module, ignored_modules, ignored_params)
        module.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        module.register_forward_hook(self.forward_post_hook)
        if 'device_id' in kwargs:
            if kwargs['device_id'] is not None:
                device_id = kwargs['device_id']
                if isinstance(device_id, torch.device) and device_id.type == 'cpu':
                    kwargs['device_ids'] = None
                else:
                    kwargs['device_ids'] = [device_id]
            else:
                kwargs['device_ids'] = None
            kwargs.pop('device_id')
        self._ddp = DistributedDataParallel(self._param_list, **kwargs)
        replicate.state(self.module)._ddp_weakref = weakref.ref(self._ddp)

    def forward_pre_hook(self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        return self._ddp._pre_forward(*args, **kwargs)

    def forward_post_hook(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        return self._ddp._post_forward(output)