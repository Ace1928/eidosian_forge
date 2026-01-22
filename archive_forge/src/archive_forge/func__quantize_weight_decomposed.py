import torch
import typing
def _quantize_weight_decomposed(weight: torch.Tensor, weight_qscheme: torch.qscheme, weight_dtype: torch.dtype, weight_scale: torch.Tensor, weight_zero_point: torch.Tensor, weight_axis: int, weight_quant_min: typing.Optional[int], weight_quant_max: typing.Optional[int]) -> torch.Tensor:
    _DTYPE_TO_QVALUE_BOUNDS = {torch.uint8: (0, 255), torch.int8: (-128, 127), torch.int32: (-2 ** 31, 2 ** 31 - 1)}
    _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE = {torch.quint8: torch.uint8, torch.qint8: torch.int8, torch.qint32: torch.int32}
    if weight_qscheme == torch.per_tensor_affine:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
            if weight_quant_min is None or weight_quant_max is None:
                weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[weight_dtype_]
            weight = torch.ops.quantized_decomposed.quantize_per_tensor(weight, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, weight_dtype_)
            return weight
    elif weight_qscheme in [torch.per_channel_affine, torch.per_channel_affine_float_qparams]:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
            if weight_quant_min is None or weight_quant_max is None:
                weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[weight_dtype_]
            weight = torch.ops.quantized_decomposed.quantize_per_channel(weight, weight_scale, weight_zero_point, weight_axis, weight_quant_min, weight_quant_max, weight_dtype_)
            return weight
    raise Exception(f'Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}')