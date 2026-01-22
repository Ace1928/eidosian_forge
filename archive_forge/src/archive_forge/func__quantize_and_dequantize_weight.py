import torch
import typing
def _quantize_and_dequantize_weight(weight: torch.Tensor, weight_qscheme: torch.qscheme, weight_dtype: torch.dtype, weight_scale: torch.Tensor, weight_zero_point: torch.Tensor, weight_axis_int: int) -> torch.Tensor:
    """ Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme in [torch.per_tensor_affine, torch.per_channel_affine, torch.per_channel_affine_float_qparams]:
        weight_quant = _quantize_weight(weight, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis_int)
        weight_dequant = weight_quant.dequantize()
    else:
        weight_dequant = weight
    return weight_dequant