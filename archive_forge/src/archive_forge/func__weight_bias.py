from typing import Optional
import torch
import torch.ao.nn.intrinsic as nni
from torch.ao.nn.sparse.quantized import linear
from torch.ao.nn.sparse.quantized.utils import LinearBlockSparsePattern
from torch.ao.nn.quantized.modules.utils import _quantize_weight, _hide_packed_params_repr
def _weight_bias(self):
    return self._packed_params._weight_bias()