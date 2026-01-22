import warnings
import sys
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import Tuple, Union, List, Optional, cast, TYPE_CHECKING
from . import _functional_collectives_impl as fun_col_impl
from ._functional_collectives_impl import _register_tensor_wrapper
from torch.fx.experimental.proxy_tensor import (
from torch._custom_ops import impl_abstract
from torch.distributed.distributed_c10d import (
def _are_we_tracing() -> bool:
    if is_torchdynamo_compiling():
        return True
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL) is not None:
        return True
    mode = get_innermost_proxy_mode()
    if mode is None:
        return False
    return mode.tracer is not None