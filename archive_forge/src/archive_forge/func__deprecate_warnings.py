import functools
import warnings
from typing import Callable, Optional, Tuple, Union
import torch
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import _mesh_resources
def _deprecate_warnings(func_name: str, extra_msg: str) -> None:
    """
    Inject common validation logics for `_prepare_input` funcs via this decorator.

    Include verifying that input needs to be either a :class:`Tensor` or :class:`DTensor`
    and only 1D :class:`DeviceMesh` is passed in.
    """
    if not is_torchdynamo_compiling():
        warnings.warn(f'{func_name} is deprecated and will be removed soon. {extra_msg}')