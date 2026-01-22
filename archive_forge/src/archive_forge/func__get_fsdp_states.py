imports. For brevity, we may import the file as ``traversal_utils``.
import collections
from typing import Deque, List, Set, Tuple
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.fsdp._common_utils import _FSDPState, _get_module_fsdp_state
def _get_fsdp_states(module: nn.Module) -> List[_FSDPState]:
    """See :func:`_get_fsdp_states_with_modules`."""
    fsdp_states, _ = _get_fsdp_states_with_modules(module)
    return fsdp_states