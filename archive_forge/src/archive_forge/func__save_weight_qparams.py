import torch
import typing
def _save_weight_qparams(destination, prefix, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis):
    destination[prefix + 'weight_qscheme'] = weight_qscheme
    destination[prefix + 'weight_dtype'] = weight_dtype
    if weight_qscheme is not None:
        destination[prefix + 'weight_scale'] = weight_scale
        destination[prefix + 'weight_zero_point'] = weight_zero_point
        if weight_qscheme == torch.per_channel_affine:
            destination[prefix + 'weight_axis'] = weight_axis