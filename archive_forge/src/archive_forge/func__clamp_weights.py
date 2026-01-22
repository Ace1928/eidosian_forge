import abc
import torch
import itertools
import collections
from torch.nn.modules.module import _addindent
def _clamp_weights(qweight, observer, scale, zp):
    if not _needs_weight_clamping(observer, qweight.dtype):
        return qweight
    observer = _get_weight_observer(observer)
    min_, max_ = (observer.quant_min, observer.quant_max)
    qw_int_max = torch.clone(qweight.int_repr()).fill_(max_)
    qw_int_min = torch.clone(qweight.int_repr()).fill_(min_)
    qw_int = torch.minimum(torch.maximum(qweight.int_repr(), qw_int_min), qw_int_max)
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch._make_per_tensor_quantized_tensor(qw_int, scale.item(), zp.item())
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine, torch.per_channel_affine_float_qparams]:
        qweight = torch._make_per_channel_quantized_tensor(qw_int, scale, zp, axis=observer.ch_axis)
    else:
        raise ValueError('Unexpected qscheme ' + observer.qscheme)
    return qweight