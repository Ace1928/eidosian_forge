from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
def _cat(tensors: List[Tensor], dim: int=0) -> Tensor:
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)