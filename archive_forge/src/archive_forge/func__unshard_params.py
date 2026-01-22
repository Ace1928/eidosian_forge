import contextlib
import warnings
from typing import cast, Generator
import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParamHandle
@contextlib.contextmanager
def _unshard_params(module: nn.Module, recurse: bool, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    """
    This unshards FSDP-managed parameters for all modules with FSDP applied in
    the module tree rooted at ``module``.
    """
    root_fsdp_states, root_fsdp_modules = _get_fsdp_root_states_with_modules(module)
    with contextlib.ExitStack() as stack:
        for root_fsdp_state, root_fsdp_module in zip(root_fsdp_states, root_fsdp_modules):
            stack.enter_context(_unshard_params_recurse(module=root_fsdp_module, state=root_fsdp_state, recurse=recurse, writeback=writeback, rank0_only=rank0_only, offload_to_cpu=offload_to_cpu, with_grads=with_grads))
        yield
    return