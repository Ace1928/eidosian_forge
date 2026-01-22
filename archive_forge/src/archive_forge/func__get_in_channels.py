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
def _get_in_channels(self, input: Tensor) -> int:
    num_spatial_dims = self._get_num_spatial_dims()
    num_dims_no_batch = num_spatial_dims + 1
    num_dims_batch = num_dims_no_batch + 1
    if input.dim() not in (num_dims_no_batch, num_dims_batch):
        raise RuntimeError('Expected {}D (unbatched) or {}D (batched) input to {}, but got input of size: {}'.format(num_dims_no_batch, num_dims_batch, self.__class__.__name__, input.shape))
    return input.shape[1] if input.dim() == num_dims_batch else input.shape[0]