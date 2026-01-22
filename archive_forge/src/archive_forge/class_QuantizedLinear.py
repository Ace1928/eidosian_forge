import warnings
from typing import List, Optional, Tuple
import torch
from torch import _VF, Tensor  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
class QuantizedLinear(torch.jit.ScriptModule):
    __constants__ = ['scale', 'zero_point']

    def __init__(self, other):
        super().__init__()
        warnings.warn('torch.jit.QuantizedLinear is deprecated and will be removed in an upcoming PyTorch release. Please use the torch.ao.nn.quantized.dynamic.Linear instead.')
        self.in_features = other.in_features
        self.out_features = other.out_features
        self.weight, self.col_offsets, self.scale, self.zero_point = torch.fbgemm_linear_quantize_weight(other.weight.clone(memory_format=torch.contiguous_format).float())
        self.weight = torch.nn.Parameter(self.weight, requires_grad=False)
        self.col_offsets = torch.nn.Parameter(self.col_offsets, requires_grad=False)
        assert other.bias is not None, 'QuantizedLinear requires a bias'
        self.bias = torch.nn.Parameter(other.bias.clone(memory_format=torch.contiguous_format).float(), requires_grad=False)
        self.register_buffer('packed_tensor_ptr', torch.fbgemm_pack_quantized_matrix(self.weight.clone(memory_format=torch.contiguous_format)))

    @torch.jit.script_method
    def _unpack(self):
        self.packed_tensor_ptr.set_(torch.fbgemm_pack_quantized_matrix(self.weight))

    @torch.jit.script_method
    def _pack(self):
        self.packed_tensor_ptr.set_(torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())

    @torch.jit.script_method
    def forward(self, input):
        out = torch.fbgemm_linear_int8_weight_fp32_activation(input.float(), self.weight, self.packed_tensor_ptr, self.col_offsets, self.scale, self.zero_point, self.bias)
        return out.to(input.dtype)

    def extra_repr(self):
        repr = 'in_features={in_features}, out_features={out_features}, scale={scale}, zero_point={zero_point}'.format(**self.__dict__)
        return repr