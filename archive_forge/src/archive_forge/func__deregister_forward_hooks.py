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
def _deregister_forward_hooks(self):
    for forward_hook_handles in self._module_to_forward_hook_handles.values():
        forward_hook_handles[0].remove()
        forward_hook_handles[1].remove()
    self._module_to_forward_hook_handles.clear()