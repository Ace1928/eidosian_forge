import warnings
from typing import List, Optional, Tuple
import torch
from torch import _VF, Tensor  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
class QuantizedLinearFP16(torch.jit.ScriptModule):

    def __init__(self, other):
        super().__init__()
        warnings.warn('torch.jit.QuantizedLinearFP16 is deprecated and will be removed in an upcoming PyTorch release. Please use the torch.ao.nn.quantized.dynamic.Linear instead.')
        self.in_features = other.in_features
        self.out_features = other.out_features
        self.original_weight = other.weight
        self.weight = torch.fbgemm_pack_gemm_matrix_fp16(other.weight.clone(memory_format=torch.contiguous_format).float())
        assert other.bias is not None, 'QuantizedLinearFP16 requires a bias'
        self.bias = torch.nn.Parameter(other.bias.clone(memory_format=torch.contiguous_format).float(), requires_grad=False)
        self.register_buffer('packed_weight', self.weight)

    @torch.jit.script_method
    def _unpack(self):
        self.packed_weight.set_(torch.fbgemm_pack_gemm_matrix_fp16(self.original_weight))

    @torch.jit.script_method
    def _pack(self):
        self.packed_weight.set_(torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())

    @torch.jit.script_method
    def forward(self, input):
        out = torch.fbgemm_linear_fp16_weight_fp32_activation(input.float(), self.packed_weight, self.bias)
        return out

    def extra_repr(self):
        repr = 'in_features={in_features}, out_features={out_features}, '.format(**self.__dict__)
        return repr