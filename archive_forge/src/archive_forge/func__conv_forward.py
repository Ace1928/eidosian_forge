import math
import warnings
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from .. import functional as F
from .. import init
from .lazy import LazyModuleMixin
from .module import Module
from .utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes
from ..common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    if self.padding_mode != 'zeros':
        return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias, self.stride, _triple(0), self.dilation, self.groups)
    return F.conv3d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)