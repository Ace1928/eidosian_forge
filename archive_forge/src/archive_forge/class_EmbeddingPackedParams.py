import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Optional, List  # noqa: F401
from .utils import _hide_packed_params_repr
from .utils import _quantize_weight
class EmbeddingPackedParams(torch.nn.Module):
    _version = 1

    def __init__(self, num_embeddings, embedding_dim, dtype=torch.quint8):
        super().__init__()
        self.dtype = dtype
        if self.dtype in [torch.quint8, torch.quint4x2]:
            scales = torch.ones(num_embeddings, dtype=torch.float)
            zero_points = torch.zeros(num_embeddings, dtype=torch.float)
            wq = torch._empty_per_channel_affine_quantized([num_embeddings, embedding_dim], scales=scales, zero_points=zero_points, axis=0, dtype=self.dtype)
            self.set_weight(wq)
        else:
            raise NotImplementedError(f'Unsupported dtype on quantized embedding! Supports quint8 and quint4x2. Got dtype: {dtype}')

    @torch.jit.export
    def set_weight(self, weight: torch.Tensor) -> None:
        if self.dtype in [torch.quint8, torch.quint4x2]:
            self._packed_weight = torch.ops.quantized.embedding_bag_prepack(weight)
        else:
            raise NotImplementedError('Unsupported dtype for quantized embedding prepack! Supports quint8 and quint4x2.')

    @torch.jit.export
    def _weight(self):
        if self.dtype in [torch.quint8, torch.quint4x2]:
            return torch.ops.quantized.embedding_bag_unpack(self._packed_weight)
        else:
            raise NotImplementedError('Unsupported dtype for quantized embedding unpack! Supports quint8 and quint4x2.')

    def forward(self, x):
        return x

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'dtype'] = self.dtype
        destination[prefix + '_packed_weight'] = self._weight()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.dtype = state_dict[prefix + 'dtype']
        state_dict.pop(prefix + 'dtype')
        weight = state_dict[prefix + '_packed_weight']
        state_dict.pop(prefix + '_packed_weight')
        self.set_weight(weight)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return self._weight().__repr__()