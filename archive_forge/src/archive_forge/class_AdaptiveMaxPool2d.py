from typing import List, Optional
from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F
from ..common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
class AdaptiveMaxPool2d(_AdaptiveMaxPoolNd):
    """Applies a 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`H_{out} \\times W_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form :math:`H_{out} \\times W_{out}`.
                     Can be a tuple :math:`(H_{out}, W_{out})` or a single :math:`H_{out}` for a
                     square image :math:`H_{out} \\times H_{out}`. :math:`H_{out}` and :math:`W_{out}`
                     can be either a ``int``, or ``None`` which means the size will be the same as that
                     of the input.
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool2d. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          :math:`(H_{out}, W_{out})=\\text{output\\_size}`.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveMaxPool2d((5, 7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveMaxPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveMaxPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)

    """
    output_size: _size_2_opt_t

    def forward(self, input: Tensor):
        return F.adaptive_max_pool2d(input, self.output_size, self.return_indices)