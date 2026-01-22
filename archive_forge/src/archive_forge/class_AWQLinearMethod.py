from typing import Any, Dict, List, Optional
import torch
from torch.nn.parameter import Parameter
from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(self, input_size_per_partition: int, output_size_per_partition: int, input_size: int, output_size: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError('The input size is not aligned with the quantized weight shape. This can be caused by too large tensor parallel size.')
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError('The output size is not aligned with the quantized weight shape. This can be caused by too large tensor parallel size.')
        qweight = Parameter(torch.empty(input_size_per_partition, output_size_per_partition // self.quant_config.pack_factor, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(qweight, {'input_dim': 0, 'output_dim': 1, 'packed_dim': 1, 'pack_factor': self.quant_config.pack_factor})
        qzeros = Parameter(torch.empty(input_size_per_partition // self.quant_config.group_size, output_size_per_partition // self.quant_config.pack_factor, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(qzeros, {'input_dim': 0, 'output_dim': 1, 'packed_dim': 1, 'pack_factor': self.quant_config.pack_factor})
        scales = Parameter(torch.empty(input_size_per_partition // self.quant_config.group_size, output_size_per_partition, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(scales, {'input_dim': 0, 'output_dim': 1})
        return {'qweight': qweight, 'qzeros': qzeros, 'scales': scales}

    def apply_weights(self, weights: Dict[str, Any], x: torch.Tensor, bias: Optional[torch.Tensor]=None) -> torch.Tensor:
        qweight = weights['qweight']
        scales = weights['scales']
        qzeros = weights['qzeros']
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256
        if FP16_MATMUL_HEURISTIC_CONDITION:
            out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
            out = torch.matmul(reshaped_x, out)
        else:
            out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)