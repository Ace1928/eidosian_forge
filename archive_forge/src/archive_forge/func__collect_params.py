import weakref
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.nn.parallel import DistributedDataParallel
from .contract import _get_registry, contract
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