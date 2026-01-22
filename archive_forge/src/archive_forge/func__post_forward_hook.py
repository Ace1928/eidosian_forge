from collections import defaultdict
from itertools import chain
import pickle
from typing import (
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode
def _post_forward_hook(module: nn.Module, inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]) -> None:
    if hasattr(module, '_memory_tracker_is_root') and module._memory_tracker_is_root:
        self._add_marker('fw_bw_boundary')