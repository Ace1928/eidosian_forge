from typing import List, Optional, Sequence, Tuple, Union
import torch
from .common import _get_storage_base
def unbind(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, ...]:
    """
    Does exactly the same as :attr:`torch.unbind` for the forward.
    In backward, avoids a :attr:`torch.cat` if the gradients
    are already multiple views of the same storage
    """
    return _Unbind.apply(x, dim)