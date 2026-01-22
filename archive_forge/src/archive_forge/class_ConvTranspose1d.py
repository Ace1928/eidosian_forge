import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch._ops import ops
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single, _pair, _triple
from torch.ao.nn.quantized.modules.conv import _reverse_repeat_padding
import torch.ao.nn.quantized as nnq
import warnings
class ConvTranspose1d(nnq.ConvTranspose1d):
    """A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose1d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nndq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nndq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nndq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nndq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """
    _FLOAT_MODULE = nn.ConvTranspose1d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        warnings.warn('The current implementation of the {} module has poor numerical accuracy and its use is not recommended'.format(self._get_name()))
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'DynamicQuantizedConvTranspose1d'

    def forward(self, input: Tensor, reduce_range: bool=True) -> Tensor:
        if len(input.shape) != 3:
            raise ValueError('Input shape must be `(N, C, L)`!')
        return torch.ops.quantized.conv_transpose1d_dynamic(input, self._packed_params, reduce_range)