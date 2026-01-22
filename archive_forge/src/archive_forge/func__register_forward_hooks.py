import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
def _register_forward_hooks(self):
    if self.mods is None:
        return
    for mod in self.mods:
        prefix = type(mod).__name__
        for name, module in dict(mod.named_modules()).items():
            if name == '':
                name = prefix
            else:
                name = '.'.join([prefix, name])
            forward_pre_hook_handle = module.register_forward_pre_hook(self._enter_module(name))
            forward_hook_handle = module.register_forward_hook(self._exit_module(name))
            self._module_to_forward_hook_handles[module] = _ForwardHookHandles(forward_pre_hook_handle, forward_hook_handle)