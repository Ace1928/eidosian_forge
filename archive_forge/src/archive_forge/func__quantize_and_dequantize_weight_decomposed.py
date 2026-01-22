import torch
import typing
def _quantize_and_dequantize_weight_decomposed(weight: torch.Tensor, weight_qscheme: torch.qscheme, weight_dtype: torch.dtype, weight_scale: torch.Tensor, weight_zero_point: torch.Tensor, weight_axis_int: int, weight_quant_min: typing.Optional[int], weight_quant_max: typing.Optional[int]) -> torch.Tensor:
    """ Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme in [torch.per_tensor_affine, torch.per_channel_affine, torch.per_channel_affine_float_qparams]:
        weight_quant = _quantize_weight_decomposed(weight, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis_int, weight_quant_min, weight_quant_max)
        weight_dequant = _dequantize_weight_decomposed(weight_quant, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis_int, weight_quant_min, weight_quant_max)
    else:
        weight_dequant = weight
    return weight_dequant