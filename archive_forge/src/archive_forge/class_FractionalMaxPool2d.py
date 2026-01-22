from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class FractionalMaxPool2d(Module):
    """Applies a 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \\times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    .. note:: Exactly one of ``output_size`` or ``output_ratio`` must be defined.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number k (for a square kernel of k x k) or a tuple `(kh, kw)`
        output_size: the target output size of the image of the form `oH x oW`.
                     Can be a tuple `(oH, oW)` or a single number oH for a square image `oH x oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :meth:`nn.MaxUnpool2d`. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          :math:`(H_{out}, W_{out})=\\text{output\\_size}` or
          :math:`(H_{out}, W_{out})=\\text{output\\_ratio} \\times (H_{in}, W_{in})`.

    Examples:
        >>> # pool of square window of size=3, and target output size 13x12
        >>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _Fractional MaxPooling:
        https://arxiv.org/abs/1412.6071
    """
    __constants__ = ['kernel_size', 'return_indices', 'output_size', 'output_ratio']
    kernel_size: _size_2_t
    return_indices: bool
    output_size: _size_2_t
    output_ratio: _ratio_2_t

    def __init__(self, kernel_size: _size_2_t, output_size: Optional[_size_2_t]=None, output_ratio: Optional[_ratio_2_t]=None, return_indices: bool=False, _random_samples=None) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.return_indices = return_indices
        self.register_buffer('_random_samples', _random_samples)
        self.output_size = _pair(output_size) if output_size is not None else None
        self.output_ratio = _pair(output_ratio) if output_ratio is not None else None
        if output_size is None and output_ratio is None:
            raise ValueError('FractionalMaxPool2d requires specifying either an output size, or a pooling ratio')
        if output_size is not None and output_ratio is not None:
            raise ValueError('only one of output_size and output_ratio may be specified')
        if self.output_ratio is not None:
            if not (0 < self.output_ratio[0] < 1 and 0 < self.output_ratio[1] < 1):
                raise ValueError(f'output_ratio must be between 0 and 1 (got {output_ratio})')

    def forward(self, input: Tensor):
        return F.fractional_max_pool2d(input, self.kernel_size, self.output_size, self.output_ratio, self.return_indices, _random_samples=self._random_samples)