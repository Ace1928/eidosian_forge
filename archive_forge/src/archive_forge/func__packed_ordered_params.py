from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
def _packed_ordered_params(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    assert self.w12 is not None, 'Packed weights are only available when using w12'
    'Used for testing - returns ordered arguments for packed operators'
    w1w2 = self.w12.weight
    b1b2_param = self.w12.bias
    w1w2 = w1w2.view([2, w1w2.shape[0] // 2, w1w2.shape[1]])
    b1b2: Optional[torch.Tensor] = None
    if b1b2_param is not None:
        b1b2 = b1b2_param.view([2, b1b2_param.shape[0] // 2])
    return (w1w2, b1b2, self.w3.weight, self.w3.bias)