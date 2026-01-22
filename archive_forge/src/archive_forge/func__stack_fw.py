from typing import List, Optional, Sequence, Tuple, Union
import torch
from .common import _get_storage_base
def _stack_fw(tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int) -> torch.Tensor:
    out = _stack_or_none_fw(tensors, dim)
    if out is None:
        out = torch.stack(tensors, dim=dim)
    return out