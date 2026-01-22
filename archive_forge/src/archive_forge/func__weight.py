import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Optional, List  # noqa: F401
from .utils import _hide_packed_params_repr
from .utils import _quantize_weight
@torch.jit.export
def _weight(self):
    if self.dtype in [torch.quint8, torch.quint4x2]:
        return torch.ops.quantized.embedding_bag_unpack(self._packed_weight)
    else:
        raise NotImplementedError('Unsupported dtype for quantized embedding unpack! Supports quint8 and quint4x2.')